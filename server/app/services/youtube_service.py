import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import requests
from pymongo import MongoClient
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from embedding_service import EmbeddingService
from aiolimiter import AsyncLimiter
import os
from datetime import datetime

from langchain_mongodb import MongoDBAtlasVectorSearch

logger = logging.getLogger(__name__)

load_dotenv()


MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
ATLAS_VECTOR_SEARCH_INDEX_NAME = os.getenv("ATLAS_VECTOR_SEARCH_INDEX_NAME")


class MongoDBEmbeddingSaver:
    def __init__(self):
        
        
        if not MONGODB_ATLAS_CLUSTER_URI:
            raise ValueError("MONGODB_URI environment variable is required")
        
        
        self.client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService()
        
        # Initialize vector store
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embedding_service,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
            relevance_score_fn="cosine"
        )
        
        print("MongoDB Vector Store initialized successfully!")
    
    async def save_video_embeddings(self, processed_video: 'ProcessedVideo') -> Dict[str, any]:
        """Save all video embeddings to MongoDB"""
        try:
            saved_documents = []
            
            # Save video metadata document
            metadata_doc = {
                "video_id": processed_video.video_info.video_id,
                "title": processed_video.video_info.title,
                "author": processed_video.video_info.author,
                "description": processed_video.video_info.description,
                "duration": processed_video.video_info.duration,
                "view_count": processed_video.video_info.view_count,
                "publish_date": processed_video.video_info.publish_date,
                "thumbnail_url": processed_video.video_info.thumbnail_url,
                "tags": processed_video.video_info.tags,
                "category": processed_video.video_info.category,
                "document_type": "video_metadata",
                "text_content": self._create_metadata_text(processed_video.video_info),
                "embedding": processed_video.metadata_embedding,
                "created_at": datetime.utcnow(),
                "full_text": processed_video.captions_text,
                "total_segments": len(processed_video.caption_segments)
            }
            
            # Insert metadata document
            metadata_result = self.collection.insert_one(metadata_doc)
            saved_documents.append({
                "type": "metadata",
                "id": str(metadata_result.inserted_id),
                "video_id": processed_video.video_info.video_id
            })
            
            # Save full text embedding document
            if processed_video.full_text_embedding:
                full_text_doc = {
                    "video_id": processed_video.video_info.video_id,
                    "title": processed_video.video_info.title,
                    "author": processed_video.video_info.author,
                    "document_type": "full_text",
                    "text_content": processed_video.captions_text,
                    "embedding": processed_video.full_text_embedding,
                    "created_at": datetime.utcnow(),
                    "character_count": len(processed_video.captions_text),
                    "segment_count": len(processed_video.caption_segments)
                }
                
                full_text_result = self.collection.insert_one(full_text_doc)
                saved_documents.append({
                    "type": "full_text",
                    "id": str(full_text_result.inserted_id),
                    "video_id": processed_video.video_info.video_id
                })
            
            # Save individual caption segments
            if processed_video.caption_segments:
                segment_docs = []
                for i, segment in enumerate(processed_video.caption_segments):
                    segment_doc = {
                        "video_id": processed_video.video_info.video_id,
                        "title": processed_video.video_info.title,
                        "author": processed_video.video_info.author,
                        "document_type": "caption_segment",
                        "segment_index": i,
                        "text_content": segment.text,
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                        "duration": segment.duration,
                        "formatted_start": segment.formatted_start,
                        "formatted_end": segment.formatted_end,
                        "embedding": segment.embedding,
                        "created_at": datetime.utcnow(),
                        "character_count": len(segment.text)
                    }
                    segment_docs.append(segment_doc)
                
                # Bulk insert segments for efficiency
                if segment_docs:
                    segments_result = self.collection.insert_many(segment_docs)
                    for idx, inserted_id in enumerate(segments_result.inserted_ids):
                        saved_documents.append({
                            "type": "caption_segment",
                            "id": str(inserted_id),
                            "video_id": processed_video.video_info.video_id,
                            "segment_index": idx
                        })
            
            return {
                "success": True,
                "saved_documents": saved_documents,
                "total_documents": len(saved_documents),
                "metadata_saved": True,
                "full_text_saved": bool(processed_video.full_text_embedding),
                "segments_saved": len(processed_video.caption_segments)
            }
            
        except Exception as e:
            logger.error(f"Error saving embeddings to MongoDB: {e}")
            return {
                "success": False,
                "error": str(e),
                "saved_documents": saved_documents
            }
    
    def _create_metadata_text(self, video_info: 'VideoInfo') -> str:
        """Create searchable text from video metadata"""
        metadata_parts = [
            f"Title: {video_info.title}",
            f"Author: {video_info.author}",
            f"Description: {video_info.description}" if video_info.description else "",
            f"Duration: {video_info.duration} seconds" if video_info.duration else "",
            f"View count: {video_info.view_count}" if video_info.view_count else "",
            f"Publish date: {video_info.publish_date}" if video_info.publish_date else "",
            f"Tags: {', '.join(video_info.tags)}" if video_info.tags else "",
            f"Category: {video_info.category}" if video_info.category else ""
        ]
        return " | ".join([part for part in metadata_parts if part])
    
    async def search_similar_content(self, query_text: str, limit: int = 5) -> List[Dict]:
        """Search for similar content using vector search"""
        try:
            # Generate embedding for query
            query_embedding = await asyncio.to_thread(
                self.embedding_service.get_document_embedding, query_text
            )
            
            # Perform vector search
            results = self.vector_store.similarity_search_with_score(
                query_text, k=limit
            )
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                }
                for doc, score in results
            ]
            
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            return []
    
    def close(self):
        """Close MongoDB connection"""
        if hasattr(self, 'client'):
            self.client.close()


@dataclass
class VideoFormat:
    itag: int
    mime_type: str
    quality: str
    fps: Optional[int]
    resolution: Optional[str]
    video_codec: Optional[str]
    audio_codec: Optional[str]
    filesize: Optional[int]
    url: str


@dataclass
class CaptionTrack:
    language_code: str
    language_name: str
    is_auto_generated: bool
    is_translatable: bool


@dataclass
class VideoInfo:
    """Contains comprehensive video information"""
    video_id: str
    title: str
    description: str
    duration: Optional[int]
    view_count: Optional[int]
    author: str
    publish_date: Optional[str]
    thumbnail_url: str
    formats: List[VideoFormat]
    available_captions: List[CaptionTrack]
    tags: List[str]
    category: Optional[str]


@dataclass
class CaptionSegment:
    """Represents a caption segment with timing and embedding"""
    text: str
    start_time: float
    end_time: float
    duration: float
    embedding: List[float]
    formatted_start: str
    formatted_end: str


@dataclass
class ProcessedVideo:
    video_info: VideoInfo
    captions_text: str
    captions_vtt: str
    caption_segments: List[CaptionSegment]
    metadata_embedding: List[float]
    full_text_embedding: List[float]


class YouTubeVideoService:
    def __init__(self, max_concurrent_embeddings: int = 3):
        self.session = requests.Session()
        self.limiter = AsyncLimiter(max_rate=2, time_period=1)
        self.embedding_service = EmbeddingService()
        self.semaphore = asyncio.Semaphore(max_concurrent_embeddings)  # Fixed: Added semaphore
        self.mongo_saver = MongoDBEmbeddingSaver()
        self.common_languages = [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'zh-CN', 'zh-TW',
            'ar', 'hi', 'hi-IN', 'th', 'vi', 'id', 'ms', 'tl', 'sv', 'no', 'da', 'fi', 'nl', 'pl',
            'tr', 'cs', 'hu', 'ro', 'bg', 'hr', 'sk', 'sl', 'et', 'lv', 'lt', 'uk', 'he'
        ]

    def extract_video_id(self, url: str) -> Optional[str]:
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/v\/([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
            r'(?:youtube\.com\/shorts\/)([^&\n?#]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    async def get_video_metadata(self, video_id: str) -> Dict:
        metadata = {}

        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = self.session.get(oembed_url, timeout=15)
            response.raise_for_status()

            data = response.json()
            metadata.update({
                'title': data.get('title', f'Video {video_id}'),
                'author': data.get('author_name', 'Unknown Author'),
                'thumbnail_url': data.get('thumbnail_url', f'https://img.youtube.com/vi/{video_id}/maxresdefault.jpg'),
            })
        except Exception as e:
            logger.warning(f"oEmbed API failed: {e}")

        # Try YouTube's internal API for additional details
        try:
            watch_url = f"https://www.youtube.com/watch?v={video_id}"
            response = self.session.get(watch_url, timeout=15)

            if response.status_code == 200:
                html_content = response.text

                # Extract additional metadata from HTML using regex
                title_match = re.search(r'"title":"([^"]+)"', html_content)
                if title_match and 'title' not in metadata:
                    metadata['title'] = title_match.group(1).encode().decode('unicode_escape')

                # Extract view count
                view_match = re.search(r'"viewCount":"(\d+)"', html_content)
                if view_match:
                    metadata['view_count'] = int(view_match.group(1))

                # Extract duration
                duration_match = re.search(r'"lengthSeconds":"(\d+)"', html_content)
                if duration_match:
                    metadata['duration'] = int(duration_match.group(1))

                # Extract description
                desc_match = re.search(r'"shortDescription":"([^"]*)"', html_content)
                if desc_match:
                    metadata['description'] = desc_match.group(1).encode().decode('unicode_escape')

                # Extract upload date
                date_match = re.search(r'"uploadDate":"([^"]+)"', html_content)
                if date_match:
                    metadata['publish_date'] = date_match.group(1)

                # Extract tags
                tags_match = re.search(r'"keywords":\[([^\]]+)\]', html_content)
                if tags_match:
                    try:
                        tags_str = '[' + tags_match.group(1) + ']'
                        metadata['tags'] = json.loads(tags_str)
                    except:
                        metadata['tags'] = []

        except Exception as e:
            logger.warning(f"Failed to extract additional metadata: {e}")

        # Fill in defaults for missing fields
        metadata.setdefault('title', f'Video {video_id}')
        metadata.setdefault('author', 'Unknown Author')
        metadata.setdefault('description', '')
        metadata.setdefault('duration', None)
        metadata.setdefault('view_count', None)
        metadata.setdefault('publish_date', None)
        metadata.setdefault('thumbnail_url', f'https://img.youtube.com/vi/{video_id}/maxresdefault.jpg')
        metadata.setdefault('tags', [])
        metadata.setdefault('category', None)

        return metadata

    async def get_available_captions(self, video_id: str) -> List[CaptionTrack]:
        """Get list of all available caption tracks for a video"""
        caption_tracks = []

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            for transcript in transcript_list:
                caption_tracks.append(CaptionTrack(
                    language_code=transcript.language_code,
                    language_name=transcript.language,
                    is_auto_generated=transcript.is_generated,
                    is_translatable=transcript.is_translatable
                ))

        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            logger.info(f"No captions available for video {video_id}: {e}")
        except Exception as e:
            logger.warning(f"Error getting caption list: {e}")

        return caption_tracks

    async def get_video_info(self, url: str) -> VideoInfo:
        """Get comprehensive video information"""
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")

            # Get metadata and available captions concurrently
            metadata_task = self.get_video_metadata(video_id)
            captions_task = self.get_available_captions(video_id)

            metadata, available_captions = await asyncio.gather(
                metadata_task, captions_task, return_exceptions=True
            )

            # Handle exceptions from concurrent tasks
            if isinstance(metadata, Exception):
                logger.error(f"Failed to get metadata: {metadata}")
                metadata = {'title': f'Video {video_id}', 'author': 'Unknown'}

            if isinstance(available_captions, Exception):
                logger.error(f"Failed to get captions list: {available_captions}")
                available_captions = []

            return VideoInfo(
                video_id=video_id,
                title=metadata['title'],
                description=metadata['description'],
                duration=metadata['duration'],
                view_count=metadata['view_count'],
                author=metadata['author'],
                publish_date=metadata['publish_date'],
                thumbnail_url=metadata['thumbnail_url'],
                formats=[],  # We're removing pytube, so no download formats
                available_captions=available_captions,
                tags=metadata['tags'],
                category=metadata['category']
            )

        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise

    async def get_captions(
            self,
            url: str,
            languages: Optional[List[str]] = None,
            prefer_manual: bool = True,
            format_type: str = "json",
            translate_to: Optional[str] = None
    ) -> Dict[str, any]:

        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                return {"success": False, "error": "Invalid YouTube URL"}

            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            except (TranscriptsDisabled, VideoUnavailable):
                return {
                    "success": False,
                    "error": "Captions are disabled or video is unavailable for this video"
                }
            except NoTranscriptFound:
                return {
                    "success": False,
                    "error": "No captions found for this video"
                }

            if not languages:
                languages = ['en']
            selected_transcript = None
            selected_language = None

            for lang in languages:
                try:
                    if prefer_manual:
                        try:
                            selected_transcript = transcript_list.find_manually_created_transcript([lang])
                            selected_language = lang
                            break
                        except NoTranscriptFound:
                            try:
                                selected_transcript = transcript_list.find_generated_transcript([lang])
                                selected_language = lang
                                break
                            except NoTranscriptFound:
                                continue
                    else:
                        selected_transcript = transcript_list.find_transcript([lang])
                        selected_language = lang
                        break
                except NoTranscriptFound:
                    continue

            if not selected_transcript:
                available_transcripts = list(transcript_list)
                if available_transcripts:
                    manual_transcripts = [t for t in available_transcripts if not t.is_generated]
                    if manual_transcripts and prefer_manual:
                        selected_transcript = manual_transcripts[0]
                    else:
                        selected_transcript = available_transcripts[0]
                    selected_language = selected_transcript.language_code

            if not selected_transcript:
                return {
                    "success": False,
                    "error": "No suitable captions found for the requested languages"
                }

            if translate_to and translate_to != selected_language:
                try:
                    selected_transcript = selected_transcript.translate(translate_to)
                    selected_language = translate_to
                except Exception as e:
                    logger.warning(f"Translation to {translate_to} failed: {e}")

            try:
                transcript_data = selected_transcript.fetch()
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to fetch caption data: {str(e)}"
                }

            formatted_captions = []
            for item in transcript_data:
                caption_dict = {
                    'text': item.get('text', '') if isinstance(item, dict) else getattr(item, 'text', ''),
                    'start': float(item.get('start', 0.0)) if isinstance(item, dict) else float(
                        getattr(item, 'start', 0.0)),
                    'duration': float(item.get('duration', 0.0)) if isinstance(item, dict) else float(
                        getattr(item, 'duration', 0.0))
                }
                formatted_captions.append(caption_dict)

            formatted_text = None
            if format_type != "json":
                formatted_text = self._format_captions(formatted_captions, format_type)

            total_duration = max([c['start'] + c['duration'] for c in formatted_captions]) if formatted_captions else 0

            return {
                "success": True,
                "video_id": video_id,
                "language": selected_language,
                "language_name": selected_transcript.language,
                "is_auto_generated": selected_transcript.is_generated,
                "format_type": format_type,
                "total_segments": len(formatted_captions),
                "duration": total_duration,
                "subtitles": formatted_captions if format_type == "json" else None,
                "text": formatted_text if format_type != "json" else None,
                "was_translated": translate_to is not None and translate_to != selected_transcript.language_code
            }

        except Exception as e:
            logger.error(f"Error getting captions: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def process_video_with_embeddings(
            self,
            url: str,
            languages: Optional[List[str]] = None,
            embed_individual_segments: bool = True,
            save_to_db: bool = True
    ) -> Tuple[ProcessedVideo, Optional[Dict]]:
        try:
            video_info = await self.get_video_info(url)

            # Fetch all caption formats in parallel
            captions_result, vtt_result, txt_result = await asyncio.gather(
                self.get_captions(url, languages=languages, format_type="json"),
                self.get_captions(url, languages=languages, format_type="vtt"),
                self.get_captions(url, languages=languages, format_type="txt")
            )

            if not captions_result["success"]:
                raise ValueError(f"Failed to get captions: {captions_result['error']}")

            captions_text = txt_result.get("text", "") if txt_result["success"] else ""
            captions_vtt = vtt_result.get("text", "") if vtt_result["success"] else ""
            subtitle_segments = captions_result.get("subtitles", [])

            metadata_text = self._create_metadata_text(video_info)
            caption_segments = []

            if embed_individual_segments and subtitle_segments:
                print(f"Generating embeddings for {len(subtitle_segments)} caption segments...")

                async def process_segment(segment):
                    text = segment.get("text", "").strip()
                    if not text:
                        return None
                    try:
                        start_time = segment.get("start", 0.0)
                        duration = segment.get("duration", 0.0)
                        end_time = start_time + duration

                        async with self.semaphore:  
                            embedding = await asyncio.to_thread(
                                self.embedding_service.get_document_embedding, text
                            )

                        return CaptionSegment(
                            text=text,
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            embedding=embedding,
                            formatted_start=self._seconds_to_vtt_time(start_time),
                            formatted_end=self._seconds_to_vtt_time(end_time)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to embed segment '{text[:50]}...': {e}")
                        return None

                # Run segment processing with controlled concurrency
                caption_segments = await asyncio.gather(
                    *[process_segment(seg) for seg in subtitle_segments]
                )
                caption_segments = [seg for seg in caption_segments if seg is not None]

            print("Generating embedding for full caption text...")
            full_text_embedding = []
            if captions_text:
                try:
                    async with self.semaphore:
                        full_text_embedding = await asyncio.to_thread(
                            self.embedding_service.get_document_embedding, captions_text
                        )
                except Exception as e:
                    logger.warning(f"Failed to generate full text embedding: {e}")

            print("Generating embeddings for metadata...")
            metadata_embedding = []
            try:
                async with self.semaphore:
                    metadata_embedding = await asyncio.to_thread(
                        self.embedding_service.get_document_embedding, metadata_text
                    )
            except Exception as e:
                logger.warning(f"Failed to generate metadata embedding: {e}")

            processed_video = ProcessedVideo(
                video_info=video_info,
                captions_text=captions_text,
                captions_vtt=captions_vtt,
                caption_segments=caption_segments,
                metadata_embedding=metadata_embedding,
                full_text_embedding=full_text_embedding
            )

            # Save to MongoDB if requested
            save_result = None
            if save_to_db:
                print("Saving embeddings to MongoDB...")
                save_result = await self.mongo_saver.save_video_embeddings(processed_video)
                if save_result["success"]:
                    print(f"✅ Successfully saved {save_result['total_documents']} documents to MongoDB")
                else:
                    print(f"❌ Failed to save to MongoDB: {save_result.get('error')}")

            return processed_video, save_result

        except Exception as e:
            logger.error(f"Error processing video with embeddings: {e}")
            raise  # Fixed: Added missing 'e'

    def _create_metadata_text(self, video_info: VideoInfo) -> str:
        metadata_parts = [
            f"Title: {video_info.title}",
            f"Author: {video_info.author}",
            f"Description: {video_info.description}" if video_info.description else "",
            f"Duration: {video_info.duration} seconds" if video_info.duration else "",
            f"View count: {video_info.view_count}" if video_info.view_count else "",
            f"Publish date: {video_info.publish_date}" if video_info.publish_date else "",
            f"Tags: {', '.join(video_info.tags)}" if video_info.tags else "",
            f"Category: {video_info.category}" if video_info.category else ""
        ]
        return " | ".join([part for part in metadata_parts if part])

    def display_embeddings_with_timestamps(self, processed_video: ProcessedVideo, max_segments: int = 10):
        print(f"\n🕒 TIMESTAMPED EMBEDDINGS")
        print("=" * 80)

        if not processed_video.caption_segments:
            print("No caption segments with embeddings available.")
            return

        segments_to_show = min(max_segments, len(processed_video.caption_segments))

        for i, segment in enumerate(processed_video.caption_segments[:segments_to_show]):
            print(f"\n📍 Segment {i + 1}:")
            print(f"   ⏰ Time: {segment.formatted_start} --> {segment.formatted_end}")
            print(f"   ⏱️  Duration: {segment.duration:.2f}s")
            print(f"   📝 Text: \"{segment.text[:100]}{'...' if len(segment.text) > 100 else ''}\"")
            print(
                f"   🧠 Embedding: [{segment.embedding[0]:.6f}, {segment.embedding[1]:.6f}, {segment.embedding[2]:.6f}, ..., {segment.embedding[-1]:.6f}]")
            print(f"   📊 Dimensions: {len(segment.embedding)}")
            print("-" * 60)

        if len(processed_video.caption_segments) > max_segments:
            remaining = len(processed_video.caption_segments) - max_segments
            print(f"\n... and {remaining} more segments")

        print(f"\n📈 SUMMARY:")
        print(f"   Total segments: {len(processed_video.caption_segments)}")
        print(
            f"   Embedding dimensions: {len(processed_video.caption_segments[0].embedding) if processed_video.caption_segments else 0}")

        if processed_video.full_text_embedding:
            print(f"   Full text embedding dimensions: {len(processed_video.full_text_embedding)}")

        if processed_video.metadata_embedding:
            print(f"   Metadata embedding dimensions: {len(processed_video.metadata_embedding)}")

   

    async def get_all_available_captions(self, url: str) -> Dict[str, any]:
        """Get information about all available caption tracks"""
        try:
            video_id = self.extract_video_id(url)
            if not video_id:
                return {"success": False, "error": "Invalid YouTube URL"}

            available_captions = await self.get_available_captions(video_id)

            return {
                "success": True,
                "video_id": video_id,
                "available_languages": [
                    {
                        "code": track.language_code,
                        "name": track.language_name,
                        "auto_generated": track.is_auto_generated,
                        "translatable": track.is_translatable
                    }
                    for track in available_captions
                ],
                "total_tracks": len(available_captions)
            }

        except Exception as e:
            logger.error(f"Error getting available captions: {e}")
            return {"success": False, "error": str(e)}

    def _format_captions(self, captions: List[Dict], format_type: str) -> str:
        """Format captions into requested format"""
        if format_type == "txt":
            return "\n".join([c.get("text", "") for c in captions])
        elif format_type == "srt":
            return self._format_as_srt(captions)
        elif format_type == "vtt":
            return self._format_as_vtt(captions)
        else:
            return ""

    def _format_as_srt(self, captions: List[Dict]) -> str:
        """Format captions as SRT"""
        srt_content = []
        for i, caption in enumerate(captions, 1):
            start = self._seconds_to_srt_time(caption.get("start", 0))
            end = self._seconds_to_srt_time(caption.get("start", 0) + caption.get("duration", 0))
            srt_content.append(f"{i}\n{start} --> {end}\n{caption.get('text', '')}\n")
        return "\n".join(srt_content)

    def _format_as_vtt(self, captions: List[Dict]) -> str:
        """Format captions as VTT"""
        vtt_content = ["WEBVTT\n"]
        for caption in captions:
            start = self._seconds_to_vtt_time(caption.get("start", 0))
            end = self._seconds_to_vtt_time(caption.get("start", 0) + caption.get("duration", 0))
            vtt_content.append(f"{start} --> {end}\n{caption.get('text', '')}\n")
        return "\n".join(vtt_content)

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to VTT time format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    async def search_videos(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar videos using vector search"""
        return await self.mongo_saver.search_similar_content(query, limit)

    def close(self):
        """Close all connections"""
        if hasattr(self, 'mongo_saver'):
            self.mongo_saver.close()
        if hasattr(self, 'session'):
            self.session.close()


# Utility functions for MongoDB operations
async def search_videos_by_content(query: str, limit: int = 2) -> List[Dict]:
    """Standalone function to search videos by content"""
    saver = MongoDBEmbeddingSaver()
    try:
        results = await saver.search_similar_content(query, limit)
        return results
    finally:
        saver.close()


async def get_video_by_id(video_id: str) -> Optional[Dict]:
    """Get video data from MongoDB by video ID"""
    try:
        client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        video_data = collection.find_one({"video_id": video_id})
        client.close()
        
        return video_data
    except Exception as e:
        logger.error(f"Error fetching video {video_id}: {e}")
        return None


async def main():
    """Main function to test the YouTube video processing and MongoDB storage"""
    test_urls = [
        "https://www.youtube.com/watch?v=GzrULKF4jk8"
    ]

    service = YouTubeVideoService()

    try:
        for url in test_urls:
            print(f"\n{'=' * 60}")
            print(f"Testing URL: {url}")
            print('=' * 60)

            try:
                # Process video with embeddings and save to MongoDB
                processed_video, save_result = await service.process_video_with_embeddings(
                    url, 
                    languages=['en'],
                    save_to_db=True
                )

                print(f"\n📹 Video Info:")
                print(f"   Title: {processed_video.video_info.title}")
                print(f"   Author: {processed_video.video_info.author}")
                print(f"   Duration: {processed_video.video_info.duration} seconds")
                print(f"   View Count: {processed_video.video_info.view_count}")

                print(f"\n📝 Captions Info:")
                print(f"   Full Text Length: {len(processed_video.captions_text)} characters")
                print(f"   VTT Length: {len(processed_video.captions_vtt)} characters")
                print(f"   Individual Segments: {len(processed_video.caption_segments)}")

                # Display embeddings
                service.display_embeddings_with_timestamps(processed_video, max_segments=5)

                # Show MongoDB save results
                if save_result:
                    print(f"\n💾 MongoDB SAVE RESULTS:")
                    print("=" * 40)
                    print(f"   Success: {save_result['success']}")
                    print(f"   Total documents saved: {save_result.get('total_documents', 0)}")
                    print(f"   Metadata saved: {save_result.get('metadata_saved', False)}")
                    print(f"   Full text saved: {save_result.get('full_text_saved', False)}")
                    print(f"   Segments saved: {save_result.get('segments_saved', 0)}")
                    
                    if not save_result['success']:
                        print(f"   Error: {save_result.get('error', 'Unknown error')}")

                # Export JSON demo
                print(f"\n💾 JSON EXPORT DEMO:")
                print("=" * 30)
                json_data = service.export_embeddings_json(processed_video)
                print(f"JSON export length: {len(json_data)} characters")

                # Test search functionality
                print(f"\n🔍 SEARCH TEST:")
                print("=" * 20)
                search_query = processed_video.video_info.title[:50]  # Use part of title as search query
                search_results = await service.search_videos(search_query, limit=3)
                print(f"Search query: '{search_query}'")
                print(f"Found {len(search_results)} similar videos")
                
                for i, result in enumerate(search_results, 1):
                    print(f"   {i}. Score: {result.get('similarity_score', 0):.4f}")
                    print(f"      Content: {result.get('content', '')[:100]}...")

                print(f"\n✅ Successfully processed and saved video with timestamped embeddings!")

            except Exception as e:
                print(f"❌ Error processing {url}: {e}")
                logger.exception("Full error traceback:")

    finally:
        # Clean up
        service.close()

    print(f"\n{'=' * 60}")
    print("Testing complete!")


# Example usage functions
async def process_single_video(url: str, languages: List[str] = None) -> Tuple[ProcessedVideo, Dict]:
    """Process a single video and save to MongoDB"""
    service = YouTubeVideoService()
    try:
        return await service.process_video_with_embeddings(url, languages or ['en'], save_to_db=True)
    finally:
        service.close()


async def batch_process_videos(urls: List[str], languages: List[str] = None) -> List[Dict]:
    """Process multiple videos in batch"""
    service = YouTubeVideoService()
    results = []
    
    try:
        for url in urls:
            try:
                processed_video, save_result = await service.process_video_with_embeddings(
                    url, languages or ['en'], save_to_db=True
                )
                results.append({
                    "url": url,
                    "video_id": processed_video.video_info.video_id,
                    "title": processed_video.video_info.title,
                    "success": True,
                    "segments_count": len(processed_video.caption_segments),
                    "save_result": save_result
                })
            except Exception as e:
                results.append({
                    "url": url,
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"Error processing {url}: {e}")
    finally:
        service.close()
    
    return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the main function
    asyncio.run(main())