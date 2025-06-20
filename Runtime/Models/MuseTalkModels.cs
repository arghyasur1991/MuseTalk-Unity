using System;
using System.Collections.Generic;
using UnityEngine;

namespace MuseTalk.Models
{
    using Utils;
    
    /// <summary>
    /// Configuration for MuseTalk inference
    /// </summary>
    [Serializable]
    public class MuseTalkConfig
    {
        public string ModelPath = "MuseTalk";
        public string Version = "v15"; // only v15 is supported
        public string Device = "cpu"; // "cpu" or "cuda"
        public int BatchSize = 4;
        public float ExtraMargin = 10f; // Additional margin for v15
        public bool UseINT8 = true; // Enable INT8 quantization (CPU-optimized, default for Mac)
        
        // Disk caching configuration
        public bool EnableDiskCache = true; // Enable persistent disk caching for avatar processing
        public string CacheDirectory = ""; // Cache directory path (empty = auto-detect)
        public int MaxCacheEntriesPerAvatar = 1000; // Maximum cache entries per avatar hash
        public long MaxCacheSizeMB = 1024; // Maximum total cache size in MB (1GB default)
        public int CacheVersionNumber = 1; // Cache version for invalidation on format changes
        public bool CacheLatentsOnly = false; // Cache only latents (faster) vs full avatar data (slower but complete)
        
        public MuseTalkConfig()
        {
        }
        
        public MuseTalkConfig(string modelPath, string version = "v15")
        {
            if (version != "v15")
            {
                throw new NotSupportedException("Only v15 is supported");
            }
            ModelPath = modelPath;
            Version = version;
        }
        
        /// <summary>
        /// Create configuration optimized for performance with disk caching
        /// </summary>
        public static MuseTalkConfig CreateOptimized(string modelPath = "MuseTalk")
        {
            return new MuseTalkConfig(modelPath)
            {
                EnableDiskCache = true,
                CacheLatentsOnly = false,
                MaxCacheSizeMB = 2048,
                UseINT8 = true
            };
        }
        
        /// <summary>
        /// Create configuration for development/debugging with full texture caching
        /// </summary>
        public static MuseTalkConfig CreateForDevelopment(string modelPath = "MuseTalk")
        {
            return new MuseTalkConfig(modelPath)
            {
                EnableDiskCache = true,
                CacheLatentsOnly = false, // Full texture caching for debugging
                MaxCacheSizeMB = 512, // Smaller cache for development
                UseINT8 = false // Full precision for better quality debugging
            };
        }
    }
    
    /// <summary>
    /// Input data for MuseTalk generation
    /// </summary>
    public class MuseTalkInput
    {
        public Texture2D[] AvatarTextures { get; set; }
        public AudioClip AudioClip { get; set; }
        public int BatchSize { get; set; } = 4;
        
        public MuseTalkInput(Texture2D[] avatarTextures, AudioClip audioClip)
        {
            AvatarTextures = avatarTextures ?? throw new ArgumentNullException(nameof(avatarTextures));
            AudioClip = audioClip ?? throw new ArgumentNullException(nameof(audioClip));
        }
        
        public MuseTalkInput(Texture2D avatarTexture, AudioClip audioClip) 
            : this(new[] { avatarTexture }, audioClip)
        {
        }
    }
    
    /// <summary>
    /// Result from MuseTalk generation
    /// </summary>
    public class MuseTalkResult
    {
        public bool Success { get; set; }
        public string ErrorMessage { get; set; }
        public List<Texture2D> GeneratedFrames { get; set; } = new List<Texture2D>();
        public int FrameCount { get; set; }
        
        // Additional metadata
        public int ProcessedAvatarCount { get; set; }
        public int AudioFeatureCount { get; set; }
        public int BatchCount { get; set; }
    }
    
    /// <summary>
    /// Face detection and landmark data
    /// REFACTORED: Uses byte arrays for internal storage instead of Texture2D for better memory efficiency
    /// </summary>
    public class FaceData
    {
        public bool HasFace { get; set; }
        public Rect BoundingBox { get; set; }
        public Vector2[] Landmarks { get; set; }
        
        // Face texture data as byte arrays (RGB24 format)
        public byte[] CroppedFaceTextureData { get; set; }
        public int CroppedFaceWidth { get; set; }
        public int CroppedFaceHeight { get; set; }
        
        public byte[] OriginalTextureData { get; set; }
        public int OriginalWidth { get; set; }
        public int OriginalHeight { get; set; }
        
        // Face parsing mask (if enabled)
        public byte[] FaceMaskData { get; set; }
        public int FaceMaskWidth { get; set; }
        public int FaceMaskHeight { get; set; }
        
        // Cached segmentation data (computed once during avatar processing)
        public byte[] FaceLargeData { get; set; }           // Cropped face region with expansion
        public int FaceLargeWidth { get; set; }
        public int FaceLargeHeight { get; set; }
        
        public byte[] SegmentationMaskData { get; set; }    // BiSeNet segmentation mask
        public int SegmentationMaskWidth { get; set; }
        public int SegmentationMaskHeight { get; set; }
        
        public Vector4 AdjustedFaceBbox { get; set; }      // Face bbox with version-specific adjustments
        public Vector4 CropBox { get; set; }               // Expanded crop box coordinates
        
        // Precomputed blending masks (computed once during avatar processing for performance)
        public byte[] MaskSmallData { get; set; }           // Small mask cropped to face region
        public int MaskSmallWidth { get; set; }
        public int MaskSmallHeight { get; set; }
        
        public byte[] FullMaskData { get; set; }            // Full mask with small mask pasted back
        public int FullMaskWidth { get; set; }
        public int FullMaskHeight { get; set; }
        
        public byte[] BoundaryMaskData { get; set; }        // Mask with upper boundary ratio applied
        public int BoundaryMaskWidth { get; set; }
        public int BoundaryMaskHeight { get; set; }
        
        public byte[] BlurredMaskData { get; set; }         // Final blurred mask for smooth blending
        public int BlurredMaskWidth { get; set; }
        public int BlurredMaskHeight { get; set; }
        
        // Helper methods to convert to Texture2D when needed for client consumption
        public Texture2D GetCroppedFaceTexture()
        {
            if (CroppedFaceTextureData == null) return null;
            return TextureUtils.BytesToTexture2D(CroppedFaceTextureData, CroppedFaceWidth, CroppedFaceHeight);
        }
        
        public Texture2D GetOriginalTexture()
        {
            if (OriginalTextureData == null) return null;
            return TextureUtils.BytesToTexture2D(OriginalTextureData, OriginalWidth, OriginalHeight);
        }
        
        public Texture2D GetFaceLargeTexture()
        {
            if (FaceLargeData == null) return null;
            return TextureUtils.BytesToTexture2D(FaceLargeData, FaceLargeWidth, FaceLargeHeight);
        }
        
        public Texture2D GetBlurredMaskTexture()
        {
            if (BlurredMaskData == null) return null;
            return TextureUtils.BytesToTexture2D(BlurredMaskData, BlurredMaskWidth, BlurredMaskHeight);
        }
        
        // Helper method to store Texture2D as byte array
        public void SetCroppedFaceTexture(Texture2D texture)
        {
            if (texture == null)
            {
                CroppedFaceTextureData = null;
                CroppedFaceWidth = 0;
                CroppedFaceHeight = 0;
                return;
            }
            
            var (data, width, height) = TextureUtils.Texture2DToBytes(texture);
            CroppedFaceTextureData = data;
            CroppedFaceWidth = width;
            CroppedFaceHeight = height;
        }
        
        public void SetOriginalTexture(Texture2D texture)
        {
            if (texture == null)
            {
                OriginalTextureData = null;
                OriginalWidth = 0;
                OriginalHeight = 0;
                return;
            }
            
            var (data, width, height) = TextureUtils.Texture2DToBytes(texture);
            OriginalTextureData = data;
            OriginalWidth = width;
            OriginalHeight = height;
        }
        
        public void SetFaceLargeTexture(Texture2D texture)
        {
            if (texture == null)
            {
                FaceLargeData = null;
                FaceLargeWidth = 0;
                FaceLargeHeight = 0;
                return;
            }
            
            var (data, width, height) = TextureUtils.Texture2DToBytes(texture);
            FaceLargeData = data;
            FaceLargeWidth = width;
            FaceLargeHeight = height;
        }
        
        public void SetBlurredMaskTexture(Texture2D texture)
        {
            if (texture == null)
            {
                BlurredMaskData = null;
                BlurredMaskWidth = 0;
                BlurredMaskHeight = 0;
                return;
            }
            
            var (data, width, height) = TextureUtils.Texture2DToBytes(texture);
            BlurredMaskData = data;
            BlurredMaskWidth = width;
            BlurredMaskHeight = height;
        }
    }
    
    /// <summary>
    /// Processed avatar data
    /// </summary>
    public class AvatarData
    {
        public List<FaceData> FaceRegions { get; set; } = new List<FaceData>();
        public List<float[]> Latents { get; set; } = new List<float[]>();
    }
    
    /// <summary>
    /// Audio feature data from Whisper
    /// Each chunk contains flattened features: [time_steps × layers × features] = [10 × 5 × 384] = [19200]
    /// This matches the Python MuseTalk chunking format exactly
    /// </summary>
    public class AudioFeatures
    {
        public List<float[]> FeatureChunks { get; set; } = new List<float[]>();
        public int SampleRate { get; set; }
        public float Duration { get; set; }
        public int ChunkCount => FeatureChunks.Count;
    }
    
    /// <summary>
    /// Blending and post-processing options
    /// </summary>
    [Serializable]
    public class BlendingOptions
    {
        public bool EnableBlending = true;
        public BlendingMode Mode = BlendingMode.Jaw;
        public float BlendStrength = 1.0f;
        public bool EnableFaceSwap = false;
        public bool EnableColorCorrection = true;
        
        public enum BlendingMode
        {
            Full,
            Jaw,
            LowerHalf
        }
    }
}