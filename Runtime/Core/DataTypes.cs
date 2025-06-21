using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

namespace LiveTalk.Core
{
    internal struct Frame
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
    internal struct RGB24 // Currently unused
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
    internal enum SamplingMode
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
    internal enum MorphologyOperation
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
    internal enum BlurDirection
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
    /// Face detection and landmark data
    /// REFACTORED: Uses byte arrays for internal storage instead of Texture2D for better memory efficiency
    /// </summary>
    internal class FaceData
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
    internal class AvatarData
    {
        public List<FaceData> FaceRegions { get; set; } = new List<FaceData>();
        public List<float[]> Latents { get; set; } = new List<float[]>();
    }
    
    /// <summary>
    /// Audio feature data from Whisper
    /// Each chunk contains flattened features: [time_steps × layers × features] = [10 × 5 × 384] = [19200]
    /// This matches the Python MuseTalk chunking format exactly
    /// </summary>
    internal class AudioFeatures
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
    internal class BlendingOptions
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
    internal class SegmentationData
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

    /// <summary>
    /// Crop information matching Python crop_info
    /// </summary>
    internal class CropInfo
    {
        public Frame ImageCrop { get; set; }
        public Frame ImageCrop256x256 { get; set; }
        public Vector2[] LandmarksCrop { get; set; }
        public Vector2[] LandmarksCrop256x256 { get; set; }
        public Matrix4x4 Transform { get; set; }
        public Matrix4x4 InverseTransform { get; set; }
    }
    
    internal class ProcessSourceImageResult
    {
        public CropInfo CropInfo { get; set; }
        public Frame SrcImg { get; set; }
        public Frame MaskOri { get; set; }
        public MotionInfo XsInfo { get; set; }
        public float[,] Rs { get; set; }
        public Tensor<float> Fs { get; set; }
        public float[] Xs { get; set; }
    }

    /// <summary>
    /// Motion information extracted from face keypoints - matches Python kp_info structure
    /// </summary>
    internal class MotionInfo
    {
        public float[] Pitch { get; set; }       // Processed pitch angles
        public float[] Yaw { get; set; }         // Processed yaw angles  
        public float[] Roll { get; set; }        // Processed roll angles
        public float[] Translation { get; set; } // t: translation parameters
        public float[] Expression { get; set; }  // exp: expression deformation
        public float[] Scale { get; set; }       // scale: scaling factor
        public float[] Keypoints { get; set; }   // kp: 3D keypoints
        public float[,] RotationMatrix { get; set; } // R_d: rotation matrix (added for Python compatibility)
    }
}