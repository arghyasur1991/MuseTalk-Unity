using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

namespace MuseTalk.Core
{
    public struct Frame
    {
        public byte[] data;
        public int width;
        public int height;

        public Frame(byte[] data, int width, int height)
        {
            this.data = data;
            this.width = width;
            this.height = height;
        }        
    }

    /// <summary>
    /// RGB24 pixel struct for efficient 3-byte operations
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct RGB24 // Currently unused
    {
        public byte r;
        public byte g;
        public byte b;
        
        public RGB24(byte r, byte g, byte b)
        {
            this.r = r;
            this.g = g;
            this.b = b;
        }
    }

    /// <summary>
    /// Sampling mode for texture resizing
    /// </summary>
    public enum SamplingMode
    {
        /// <summary>
        /// Bilinear interpolation - higher quality, slower (default for ML preprocessing)
        /// </summary>
        Bilinear,
        /// <summary>
        /// Point/Nearest neighbor sampling - faster, lower quality (good for face detection)
        /// </summary>
        Point
    }

    /// <summary>
    /// Morphological operation type for consolidated morphology function
    /// </summary>
    public enum MorphologyOperation
    {
        /// <summary>
        /// Dilation - expands bright regions (finds maximum in kernel neighborhood)
        /// </summary>
        Dilation,
        /// <summary>
        /// Erosion - shrinks bright regions (finds minimum in kernel neighborhood)
        /// </summary>
        Erosion
    }

    /// <summary>
    /// Blur direction for consolidated blur pass function
    /// </summary>
    public enum BlurDirection
    {
        /// <summary>
        /// Horizontal blur - samples along X axis
        /// </summary>
        Horizontal,
        /// <summary>
        /// Vertical blur - samples along Y axis
        /// </summary>
        Vertical
    }

    /// <summary>
    /// Input for MuseTalk inference - simplified for streaming
    /// </summary>
    public class MuseTalkInput
    {
        /// <summary>
        /// Avatar images for talking head generation
        /// </summary>
        public Texture2D[] AvatarTextures { get; set; }
        
        /// <summary>
        /// Audio clip for lip sync
        /// </summary>
        public AudioClip AudioClip { get; set; }
        
        /// <summary>
        /// Batch size for processing
        /// </summary>
        public int BatchSize { get; set; } = 4;
        
        public MuseTalkInput(Texture2D avatarTexture, AudioClip audioClip)
        {
            AvatarTextures = new[] { avatarTexture ?? throw new ArgumentNullException(nameof(avatarTexture)) };
            AudioClip = audioClip ?? throw new ArgumentNullException(nameof(audioClip));
        }
        
        public MuseTalkInput(Texture2D[] avatarTextures, AudioClip audioClip)
        {
            AvatarTextures = avatarTextures ?? throw new ArgumentNullException(nameof(avatarTextures));
            AudioClip = audioClip ?? throw new ArgumentNullException(nameof(audioClip));
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
        public Frame CroppedFaceTexture { get; set; }
        
        public Frame OriginalTexture { get; set; }
        
        // Face parsing mask (if enabled)
        public Frame FaceMask { get; set; }
        
        // Cached segmentation data (computed once during avatar processing)
        public Frame FaceLarge { get; set; }           // Cropped face region with expansion
        
        public Frame SegmentationMask { get; set; }    // BiSeNet segmentation mask
        
        public Vector4 AdjustedFaceBbox { get; set; }      // Face bbox with version-specific adjustments
        public Vector4 CropBox { get; set; }               // Expanded crop box coordinates
        
        // Precomputed blending masks (computed once during avatar processing for performance)
        public Frame MaskSmall { get; set; }           // Small mask cropped to face region
        
        public Frame FullMask { get; set; }            // Full mask with small mask pasted back
        
        public Frame BoundaryMask { get; set; }        // Mask with upper boundary ratio applied
        
        public Frame BlurredMask { get; set; }         // Final blurred mask for smooth blending
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

    /// <summary>
    /// Precomputed segmentation data for efficient frame blending
    /// </summary>
    public class SegmentationData
    {
        public Frame FaceLarge { get; set; }
        
        public Frame SegmentationMask { get; set; }
        
        public Vector4 AdjustedFaceBbox { get; set; }
        public Vector4 CropBox { get; set; }
        
        // Precomputed masks for efficient blending
        public Frame MaskSmall { get; set; }
        
        public Frame FullMask { get; set; }
        
        public Frame BoundaryMask { get; set; }
        
        public Frame BlurredMask { get; set; }
    }
}