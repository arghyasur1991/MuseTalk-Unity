using System;
using System.Collections.Generic;
using UnityEngine;

namespace MuseTalk.Models
{
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
    /// </summary>
    public class FaceData
    {
        public bool HasFace { get; set; }
        public Rect BoundingBox { get; set; }
        public Vector2[] Landmarks { get; set; }
        public Texture2D CroppedFaceTexture { get; set; }
        public Texture2D OriginalTexture { get; set; }
        
        // Face parsing mask (if enabled)
        public Texture2D FaceMask { get; set; }
        
        // Cached segmentation data (computed once during avatar processing)
        public Texture2D FaceLarge { get; set; }           // Cropped face region with expansion
        public Texture2D SegmentationMask { get; set; }    // BiSeNet segmentation mask
        public Vector4 AdjustedFaceBbox { get; set; }      // Face bbox with version-specific adjustments
        public Vector4 CropBox { get; set; }               // Expanded crop box coordinates
        
        // Precomputed blending masks (computed once during avatar processing for performance)
        public Texture2D MaskSmall { get; set; }           // Small mask cropped to face region
        public Texture2D FullMask { get; set; }            // Full mask with small mask pasted back
        public Texture2D BoundaryMask { get; set; }        // Mask with upper boundary ratio applied
        public Texture2D BlurredMask { get; set; }         // Final blurred mask for smooth blending
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