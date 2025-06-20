using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MuseTalk.Core
{
    using Utils;
    using Models;

    /// <summary>
    /// Precomputed segmentation data for efficient frame blending
    /// REFACTORED: Uses byte arrays for internal storage instead of Texture2D for better memory efficiency
    /// </summary>
    public class SegmentationData
    {
        public byte[] FaceLargeData { get; set; }
        public int FaceLargeWidth { get; set; }
        public int FaceLargeHeight { get; set; }
        
        public byte[] SegmentationMaskData { get; set; }
        public int SegmentationMaskWidth { get; set; }
        public int SegmentationMaskHeight { get; set; }
        
        public Vector4 AdjustedFaceBbox { get; set; }
        public Vector4 CropBox { get; set; }
        
        // Precomputed masks for efficient blending
        public byte[] MaskSmallData { get; set; }
        public int MaskSmallWidth { get; set; }
        public int MaskSmallHeight { get; set; }
        
        public byte[] FullMaskData { get; set; }
        public int FullMaskWidth { get; set; }
        public int FullMaskHeight { get; set; }
        
        public byte[] BoundaryMaskData { get; set; }
        public int BoundaryMaskWidth { get; set; }
        public int BoundaryMaskHeight { get; set; }
        
        public byte[] BlurredMaskData { get; set; }
        public int BlurredMaskWidth { get; set; }
        public int BlurredMaskHeight { get; set; }
    }

    /// <summary>
    /// Avatar animation cache key based on texture content hashes
    /// </summary>
    public class AvatarAnimationKey
    {
        public string[] TextureHashes { get; set; }
        public string Version { get; set; }
        
        public override bool Equals(object obj)
        {
            if (obj is AvatarAnimationKey other)
            {
                return Version == other.Version && 
                       TextureHashes != null && other.TextureHashes != null &&
                       TextureHashes.Length == other.TextureHashes.Length &&
                       TextureHashes.SequenceEqual(other.TextureHashes);
            }
            return false;
        }
        
        public override int GetHashCode()
        {
            unchecked
            {
                int hash = Version?.GetHashCode() ?? 0;
                if (TextureHashes != null)
                {
                    foreach (var textureHash in TextureHashes)
                    {
                        hash = hash * 31 + (textureHash?.GetHashCode() ?? 0);
                    }
                }
                return hash;
            }
        }
    }

    /// <summary>
    /// Core MuseTalk inference engine that manages ONNX models for real-time talking head generation
    /// 
    /// MEMORY OPTIMIZATIONS:
    /// - Zero-copy tensor operations in UNet and VAE decoder pipeline
    /// - Reusable batch arrays to eliminate repeated allocations  
    /// - Direct buffer sharing between ONNX tensors without intermediate copying
    /// - Persistent ONNX result storage to prevent premature garbage collection
    /// - Avatar animation caching for repeated sequences
    /// </summary>
    public class MuseTalkInference : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        
        // ONNX Runtime sessions
        private InferenceSession _unetSession;
        private InferenceSession _vaeEncoderSession;
        private InferenceSession _vaeDecoderSession;
        private InferenceSession _positionalEncodingSession;
        
        // Whisper model for audio feature extraction
        private readonly WhisperModel _whisperModel;
        
        // Configuration
        private MuseTalkConfig _config;
        private readonly bool _initialized = false;
        private bool _disposed = false;
        
        // QUANTIZATION SUPPORT: INT8 configuration for CPU optimization
        private readonly bool _useINT8 = false;
        
        // Face analysis utility for SCRFD+1k3d68 face processing
        private readonly FaceAnalysis _faceAnalysis;
        
        // Avatar data for blending
        private AvatarData _avatarData;
        
        // Avatar animation segmentation cache
        private static readonly Dictionary<AvatarAnimationKey, AvatarData> _avatarAnimationCache = new();
        private const int MAX_CACHE_SIZE = 1000; // Limit cache size to prevent memory bloat
        
        // Disk cache for persistent avatar processing results
        private readonly AvatarDiskCache _diskCache;
        
        // Reusable batch processing array (class-level for memory reuse)
        private float[] _reusableBatchArray = new float[0];
        
        // Reusable UNet result to keep tensor memory alive across pipeline
        private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _reusableUNetResult;
        
        public bool IsInitialized => _initialized;
        public static bool LogTiming { get; set; } = false;
        
        /// <summary>
        /// Enable detailed performance monitoring and logging
        /// </summary>
        public static bool EnablePerformanceMonitoring { get; set; } = true;
        
        /// <summary>
        /// Returns true if INT8 quantization is currently active (excluding VAE models for quality)
        /// </summary>
        public bool IsUsingINT8 => _useINT8;
        
        /// <summary>
        /// Get the current quantization mode being used
        /// </summary>
        public string QuantizationMode
        {
            get
            {
                if (_useINT8) return "Optimal (INT8 for performance, FP32 for VAE quality)";
                return "FP32 only";
            }
        }
        
        /// <summary>
        /// Initialize MuseTalk inference with specified configuration
        /// </summary>
        public MuseTalkInference(MuseTalkConfig config)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            
            // QUANTIZATION: Initialize INT8 support based on config
            _useINT8 = _config.UseINT8;
            
            try
            {
                InitializeModels();
                _faceAnalysis = new FaceAnalysis(_config);
                
                // Initialize Whisper model for audio processing
                _whisperModel = new WhisperModel(_config);
                
                // Initialize disk cache for persistent avatar processing results
                try
                {
                    _diskCache = new AvatarDiskCache(_config);
                    Logger.Log("[MuseTalkInference] Disk cache initialized successfully");
                }
                catch (Exception diskCacheEx)
                {
                    Logger.LogWarning($"[MuseTalkInference] Failed to initialize disk cache: {diskCacheEx.Message}. Proceeding without disk caching.");
                    // _diskCache remains null, will be handled gracefully in code
                }
                
                _initialized = true;
                Logger.Log("[MuseTalkInference] Successfully initialized");
            }
            catch (Exception e)
            {
                Logger.LogError($"[MuseTalkInference] Failed to initialize: {e.Message}");
                _initialized = false;
            }
        }
        
        /// <summary>
        /// Initialize all ONNX models
        /// </summary>
        private void InitializeModels()
        {
            _unetSession = ModelUtils.LoadModel(_config, "unet");
            _vaeEncoderSession = ModelUtils.LoadModel(_config, "vae_encoder");
            _vaeDecoderSession = ModelUtils.LoadModel(_config, "vae_decoder");
            _positionalEncodingSession = ModelUtils.LoadModel(_config, "positional_encoding");
        }

        /// <summary>
        /// Generate talking head video frames from avatar images and audio
        /// </summary>
        public async Task<MuseTalkResult> GenerateAsync(MuseTalkInput input)
        {
            if (!_initialized)
                throw new InvalidOperationException("MuseTalk inference not initialized");
                
            if (input == null)
                throw new ArgumentNullException(nameof(input));
                       
            try
            {
                Logger.Log($"[MuseTalkInference] === STARTING MUSETALK GENERATION ===");
                Logger.Log($"[MuseTalkInference] Version: {_config.Version}, Batch Size: {input.BatchSize}");
                Logger.Log($"[MuseTalkInference] Avatar Images: {input.AvatarTextures.Length}, Audio: {input.AudioClip.name} ({input.AudioClip.length:F2}s)");
                
                // Step 1: Process avatar images and extract face regions
                Logger.Log("[MuseTalkInference] STAGE 1: Processing avatar images...");
                var avatarData = await ProcessAvatarImages(input.AvatarTextures);
                Logger.Log($"[MuseTalkInference] Stage 1 completed - Processed {avatarData.FaceRegions.Count} faces");
                
                // Step 2: Process audio and extract features
                Logger.Log("[MuseTalkInference] STAGE 2: Processing audio...");
                var audioFeatures = await ProcessAudio(input.AudioClip);
                Logger.Log($"[MuseTalkInference] Stage 2 completed - Generated {audioFeatures.FeatureChunks.Count} audio chunks");
                
                // Step 3: Generate video frames
                Logger.Log("[MuseTalkInference] STAGE 3: Generating video frames...");
                var frames = await GenerateFrames(avatarData, audioFeatures, input.BatchSize);
                Logger.Log($"[MuseTalkInference] Stage 3 completed - Generated {frames.Count} frames");
                
                var result = new MuseTalkResult
                {
                    Success = true,
                    GeneratedFrames = frames,
                    FrameCount = frames.Count,
                };
                
                Logger.Log($"[MuseTalkInference] === GENERATION COMPLETED ===");
                    
                return result;
            }
            catch (Exception e)
            {
                Logger.LogError($"[MuseTalkInference] Generation failed: {e.Message}");
                return new MuseTalkResult
                {
                    Success = false,
                    ErrorMessage = e.Message
                };
            }
        }
        
        /// <summary>
        /// Pre-compute segmentation data that can be cached and reused for all frames
        /// This includes face_large crop, BiSeNet segmentation mask, and all blending masks
        /// REFACTORED: Uses byte arrays internally for better memory efficiency
        /// </summary>
        private SegmentationData PrecomputeSegmentationData(byte[] originalImage, int originalWidth, int originalHeight, Vector4 faceBbox, string version)
        {
            // Apply version-specific adjustments to face bbox (matching BlendFaceWithOriginal logic)
            Vector4 adjustedFaceBbox = faceBbox;
            if (version == "v15") // v15 mode
            {
                // Apply v15 extra margin to y2 (bottom of face bbox)
                adjustedFaceBbox.w = Mathf.Min(adjustedFaceBbox.w + 10f, originalHeight);
            }
            
            // Calculate expanded crop box for face_large (matching Python's crop_box calculation)
            var cropRect = GetCropBox(adjustedFaceBbox, 1.5f); // expandFactor = 1.5f
            var cropBox = new Vector4(cropRect.x, cropRect.y, cropRect.x + cropRect.width, cropRect.y + cropRect.height);
            
            // Create face_large crop (matching Python's face_large = body.crop(crop_box))
            var faceLarge = CropImage(originalImage, originalWidth, originalHeight, cropRect);
            
            if (faceLarge == null)
                throw new InvalidOperationException("Failed to create face_large crop");
            
            // Generate face segmentation mask using BiSeNet on face_large
            var (segmentationMaskData, segmentationMaskWidth, segmentationMaskHeight) = GenerateFaceSegmentationMaskCached(faceLarge, (int)cropRect.width, (int)cropRect.height);
            
            if (segmentationMaskData == null)
                throw new InvalidOperationException("Failed to generate segmentation mask");
            
            // OPTIMIZATION: Precompute all blending masks that are independent of faceTexture
            // These masks only depend on segmentation, face bbox, and crop box - all available now
            
            // Convert segmentation mask byte array to texture for ImageBlendingHelper methods
            var segmentationMask = TextureUtils.BytesToTexture2D(segmentationMaskData, segmentationMaskWidth, segmentationMaskHeight);
            
            // Step 1: Create mask_small by cropping BiSeNet mask to face bbox (matching Python)
            var maskSmall = ImageBlendingHelper.CreateSmallMaskFromBiSeNet(segmentationMask, adjustedFaceBbox, cropRect);
            
            // Step 2: Create full mask by pasting mask_small back into face_large dimensions (matching Python)
            var fullMask = ImageBlendingHelper.CreateFullMask(segmentationMask, maskSmall, adjustedFaceBbox, cropRect);
            
            // Step 3: Apply upper boundary ratio to preserve upper face (matching Python)
            const float upperBoundaryRatio = 0.5f; // Standard value used in ApplySegmentationMask
            var boundaryMask = ImageBlendingHelper.ApplyUpperBoundaryRatio(fullMask, upperBoundaryRatio);
            
            // Step 4: Apply Gaussian blur for smooth blending (matching Python)
            var blurredMask = ImageBlendingHelper.ApplyGaussianBlurToMask(boundaryMask);
            
            // Convert all textures to byte arrays for efficient storage
            var (maskSmallData, maskSmallWidth, maskSmallHeight) = TextureUtils.Texture2DToBytes(maskSmall);
            var (fullMaskData, fullMaskWidth, fullMaskHeight) = TextureUtils.Texture2DToBytes(fullMask);
            var (boundaryMaskData, boundaryMaskWidth, boundaryMaskHeight) = TextureUtils.Texture2DToBytes(boundaryMask);
            var (blurredMaskData, blurredMaskWidth, blurredMaskHeight) = TextureUtils.Texture2DToBytes(blurredMask);
            
            // faceLarge is already byte array, so use it directly
            var faceLargeData = faceLarge;
            var faceLargeWidth = (int)cropRect.width;
            var faceLargeHeight = (int)cropRect.height;
            
            // Clean up temporary textures (segmentationMaskData is already byte array, so no cleanup needed for that)
            UnityEngine.Object.Destroy(segmentationMask);
            UnityEngine.Object.Destroy(maskSmall);
            UnityEngine.Object.Destroy(fullMask);
            UnityEngine.Object.Destroy(boundaryMask);
            UnityEngine.Object.Destroy(blurredMask);
            
            return new SegmentationData
            {
                FaceLargeData = faceLargeData,
                FaceLargeWidth = faceLargeWidth,
                FaceLargeHeight = faceLargeHeight,
                
                SegmentationMaskData = segmentationMaskData,
                SegmentationMaskWidth = segmentationMaskWidth,
                SegmentationMaskHeight = segmentationMaskHeight,
                
                AdjustedFaceBbox = adjustedFaceBbox,
                CropBox = cropBox,
                
                // Precomputed masks for efficient blending
                MaskSmallData = maskSmallData,
                MaskSmallWidth = maskSmallWidth,
                MaskSmallHeight = maskSmallHeight,
                
                FullMaskData = fullMaskData,
                FullMaskWidth = fullMaskWidth,
                FullMaskHeight = fullMaskHeight,
                
                BoundaryMaskData = boundaryMaskData,
                BoundaryMaskWidth = boundaryMaskWidth,
                BoundaryMaskHeight = boundaryMaskHeight,
                
                BlurredMaskData = blurredMaskData,
                BlurredMaskWidth = blurredMaskWidth,
                BlurredMaskHeight = blurredMaskHeight
            };
        }
        
        /// <summary>
        /// Generate segmentation mask for caching (extracted from ImageBlendingHelper logic)
        /// OPTIMIZED: Returns byte array directly without creating temporary Texture2D objects
        /// FIXED: Runs on main thread to avoid Unity texture operation violations
        /// </summary>
        private (byte[], int, int) GenerateFaceSegmentationMaskCached(byte[] faceLarge, int faceLargeWidth, int faceLargeHeight)
        {
            try
            {
                // Run BiSeNet directly on the face_large crop using byte array optimized method
                var maskResult = _faceAnalysis.CreateFaceMaskWithMorphology(faceLarge, faceLargeWidth, faceLargeHeight, "jaw");
                var maskData = maskResult.Item1;
                var maskWidth = maskResult.Item2;
                var maskHeight = maskResult.Item3;
                
                if (maskData != null)
                {
                    // Resize to target dimensions if needed
                    if (maskWidth != faceLargeWidth || maskHeight != faceLargeHeight)
                    {
                        var resizedMaskData = TextureUtils.ResizeTextureToExactSize(maskData, maskWidth, maskHeight, faceLargeWidth, faceLargeHeight, TextureUtils.SamplingMode.Bilinear);
                        return (resizedMaskData, faceLargeWidth, faceLargeHeight);
                    }
                    return (maskData, maskWidth, maskHeight);
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[MuseTalkInference] ONNX face parsing failed: {e.Message}");
            }
            
            throw new InvalidOperationException("Face segmentation failed and no fallback is available");
        }
        
        /// <summary>
        /// Calculate crop box with expansion factor (matching Python get_crop_box)
        /// </summary>
        private Rect GetCropBox(Vector4 faceBbox, float expandFactor)
        {
            // Python: x, y, x1, y1 = box
            float x = faceBbox.x;
            float y = faceBbox.y;
            float x1 = faceBbox.z;
            float y1 = faceBbox.w;
            
            // Python: x_c, y_c = (x+x1)//2, (y+y1)//2 (integer division!)
            int xCenter = (int)((x + x1) / 2);
            int yCenter = (int)((y + y1) / 2);
            
            // Python: w, h = x1-x, y1-y
            float width = x1 - x;
            float height = y1 - y;
            
            // Python: s = int(max(w, h)//2*expand) (integer conversion!)
            int s = (int)(Mathf.Max(width, height) / 2 * expandFactor);
            
            // Python: crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
            return new Rect(xCenter - s, yCenter - s, 2 * s, 2 * s);
        }
        
        /// <summary>
        /// Crop image to specified rectangle
        /// </summary>
        private byte[] CropImage(byte[] source, int sourceWidth, int sourceHeight, Rect cropRect)
        {
            // Ensure crop bounds are within image
            cropRect.x = Mathf.Max(0, cropRect.x);
            cropRect.y = Mathf.Max(0, cropRect.y);
            cropRect.width = Mathf.Min(cropRect.width, sourceWidth - cropRect.x);
            cropRect.height = Mathf.Min(cropRect.height, sourceHeight - cropRect.y);
            
            return TextureUtils.CropTexture(source, sourceWidth, sourceHeight, cropRect);
        }

        
        
        /// <summary>
        /// Create cache key for avatar animation sequence
        /// </summary>
        private AvatarAnimationKey CreateAvatarAnimationKey(Texture2D[] avatarTextures, string version)
        {
            var textureHashes = new string[avatarTextures.Length];
            for (int i = 0; i < avatarTextures.Length; i++)
            {
                textureHashes[i] = TextureUtils.GenerateTextureHash(avatarTextures[i]);
            }
            
            return new AvatarAnimationKey
            {
                TextureHashes = textureHashes,
                Version = version
            };
        }
        
        /// <summary>
        /// Manage cache size to prevent memory bloat
        /// </summary>
        private void ManageCacheSize()
        {
            if (_avatarAnimationCache.Count > MAX_CACHE_SIZE)
            {
                // Remove oldest entries (simple FIFO for now)
                var keysToRemove = _avatarAnimationCache.Keys.Take(_avatarAnimationCache.Count - MAX_CACHE_SIZE + 1).ToList();
                foreach (var key in keysToRemove)
                {
                    if (_avatarAnimationCache.TryGetValue(key, out var cachedData))
                    {
                        // Clean up cached byte arrays (no Texture2D cleanup needed since we use byte arrays now)
                        foreach (var faceRegion in cachedData.FaceRegions)
                        {
                            // Data is stored as byte arrays, so no Unity Object cleanup needed
                            // The garbage collector will handle the byte arrays when the cache entry is removed
                        }
                    }
                    _avatarAnimationCache.Remove(key);
                    Logger.Log($"[MuseTalkInference] Removed cached avatar animation to manage memory");
                }
            }
        }

        /// <summary>
        /// Process avatar images and extract face regions with landmarks
        /// Matches Python's get_landmark_and_bbox functionality exactly
        /// OPTIMIZED: Uses avatar animation caching for repeated sequences
        /// ENHANCED: Added disk caching for persistent avatar processing results
        /// </summary>
        private async Task<AvatarData> ProcessAvatarImages(Texture2D[] avatarTextures)
        {
            // Generate cache keys for both in-memory and disk caching
            var memoryCacheKey = CreateAvatarAnimationKey(avatarTextures, _config.Version);
            var diskCacheKey = _diskCache?.GenerateAvatarCacheKey(avatarTextures, _config.Version);
            
            // Check in-memory cache first (fastest)
            if (_avatarAnimationCache.TryGetValue(memoryCacheKey, out var cachedAvatarData))
            {
                Logger.Log($"[MuseTalkInference] Using in-memory cached avatar animation data for {avatarTextures.Length} textures - {cachedAvatarData.FaceRegions.Count} faces, {cachedAvatarData.Latents.Count} latents");
                
                // Validate cached data integrity
                if (cachedAvatarData.Latents.Count == 0)
                {
                    Logger.LogWarning("[MuseTalkInference] Cached avatar data has no latents - removing from cache and reprocessing");
                    _avatarAnimationCache.Remove(memoryCacheKey);
                }
                else
                {
                    _avatarData = cachedAvatarData;
                    return cachedAvatarData;
                }
            }
            
            // Check disk cache second (slower but persistent across sessions)
            AvatarData diskCachedData = null;
            if (_diskCache != null && !string.IsNullOrEmpty(diskCacheKey))
            {
                try
                {
                    diskCachedData = await _diskCache.TryLoadAvatarDataAsync(diskCacheKey);
                    if (diskCachedData != null && diskCachedData.Latents.Count > 0)
                    {
                        Logger.Log($"[MuseTalkInference] Using disk cached avatar data for {avatarTextures.Length} textures - {diskCachedData.FaceRegions.Count} faces, {diskCachedData.Latents.Count} latents");
                        
                        // If disk cache only has latents but missing texture data, we need to reprocess face regions
                        bool needsTextureReprocessing = false;
                        if (diskCachedData.FaceRegions.Count > 0)
                        {
                            var firstFace = diskCachedData.FaceRegions[0];
                            if (firstFace.OriginalTextureData == null || firstFace.CroppedFaceTextureData == null || 
                                firstFace.BlurredMaskData == null || firstFace.FaceLargeData == null)
                            {
                                needsTextureReprocessing = true;
                                Logger.Log("[MuseTalkInference] Disk cache has latents but missing texture data, will reprocess face regions for blending");
                            }
                        }
                        
                        if (!needsTextureReprocessing)
                        {
                            // Complete cache hit - store in memory cache for faster future access
                            ManageCacheSize();
                            _avatarAnimationCache[memoryCacheKey] = diskCachedData;
                            
                            _avatarData = diskCachedData;
                            return diskCachedData;
                        }
                        else
                        {
                            // Partial cache hit - we have latents but need to reprocess textures
                            // This will be handled after face detection by using cached latents
                            Logger.Log("[MuseTalkInference] Using cached latents but reprocessing face regions");
                        }
                    }
                }
                catch (Exception e)
                {
                    Logger.LogWarning($"[MuseTalkInference] Failed to load from disk cache: {e.Message}");
                }
            }
            
            Logger.Log($"[MuseTalkInference] Processing new avatar animation sequence with {avatarTextures.Length} textures");
            
            var avatarData = new AvatarData();

            var textures = new List<byte[]>();
            foreach (var texture in avatarTextures)
            {
                var (textureBytes, _, _) = TextureUtils.Texture2DToBytes(texture);
                textures.Add(textureBytes);
            }
            
            // Face Detection
            var result = _faceAnalysis.GetLandmarkAndBbox(
                textures, 
                avatarTextures[0].width,
                avatarTextures[0].height
            );
            List<Vector4> coordsList = result.Item1;
            List<byte[]> framesList = result.Item2;
            
            Logger.Log($"[MuseTalkInference] Face detection completed: {coordsList.Count} results for {avatarTextures.Length} input textures");
            
            // Process each detected face region
            for (int i = 0; i < coordsList.Count; i++)
            {
                var bbox = coordsList[i];
                
                if (bbox == Vector4.zero)
                {
                    Logger.LogWarning($"[MuseTalkInference] No face detected in image {i}, skipping");
                    continue;
                }
                
                try
                {                    
                    var originalTexture = framesList[i];
                    
                    // Crop face region with version-specific margins
                    int originalWidth = avatarTextures[0].width;
                    int originalHeight = avatarTextures[0].height;
                    var croppedTextureData = _faceAnalysis.CropFaceRegion(originalTexture, originalWidth, originalHeight, bbox, _config.Version);
                    
                    // Standard face crop dimensions (matching MuseTalk expectations)
                    int croppedWidth = 256;
                    int croppedHeight = 256;
                    
                    // Pre-compute segmentation mask and cached data for blending
                    var segmentationData = PrecomputeSegmentationData(originalTexture, originalWidth, originalHeight, bbox, _config.Version);
                   
                    // Create face data for this region using byte arrays
                    var faceData = new FaceData
                    {
                        HasFace = true,
                        BoundingBox = new Rect(bbox.x, bbox.y, bbox.z - bbox.x, bbox.w - bbox.y),
                        
                        // Face texture data as byte arrays
                        CroppedFaceTextureData = croppedTextureData,
                        CroppedFaceWidth = croppedWidth,
                        CroppedFaceHeight = croppedHeight,
                        
                        OriginalTextureData = originalTexture,
                        OriginalWidth = originalWidth,
                        OriginalHeight = originalHeight,
                        
                        // Cached segmentation data
                        FaceLargeData = segmentationData.FaceLargeData,
                        FaceLargeWidth = segmentationData.FaceLargeWidth,
                        FaceLargeHeight = segmentationData.FaceLargeHeight,
                        
                        SegmentationMaskData = segmentationData.SegmentationMaskData,
                        SegmentationMaskWidth = segmentationData.SegmentationMaskWidth,
                        SegmentationMaskHeight = segmentationData.SegmentationMaskHeight,
                        
                        AdjustedFaceBbox = segmentationData.AdjustedFaceBbox,
                        CropBox = segmentationData.CropBox,
                        
                        // Precomputed blending masks
                        MaskSmallData = segmentationData.MaskSmallData,
                        MaskSmallWidth = segmentationData.MaskSmallWidth,
                        MaskSmallHeight = segmentationData.MaskSmallHeight,
                        
                        FullMaskData = segmentationData.FullMaskData,
                        FullMaskWidth = segmentationData.FullMaskWidth,
                        FullMaskHeight = segmentationData.FullMaskHeight,
                        
                        BoundaryMaskData = segmentationData.BoundaryMaskData,
                        BoundaryMaskWidth = segmentationData.BoundaryMaskWidth,
                        BoundaryMaskHeight = segmentationData.BoundaryMaskHeight,
                        
                        BlurredMaskData = segmentationData.BlurredMaskData,
                        BlurredMaskWidth = segmentationData.BlurredMaskWidth,
                        BlurredMaskHeight = segmentationData.BlurredMaskHeight
                    };
                    
                    avatarData.FaceRegions.Add(faceData);
                    
                    // Get latents for UNet - use cached latents if available
                    float[] latents;
                    if (diskCachedData != null && i < diskCachedData.Latents.Count)
                    {
                        // Use cached latents from disk
                        latents = diskCachedData.Latents[i];
                        Logger.Log($"[MuseTalkInference] Using cached latents for face region {i}");
                    }
                    else
                    {
                        // Generate new latents
                        latents = await GetLatentsForUNet(croppedTextureData, croppedWidth, croppedHeight);
                    }
                    avatarData.Latents.Add(latents);
                }
                catch (Exception e)
                {
                    Logger.LogError($"[MuseTalkInference] Error processing face region {i}: {e.Message}");
                    Logger.LogError($"[MuseTalkInference] Stack trace: {e.StackTrace}");
                    // Continue processing other faces, but this will be caught in validation later
                }
            }
            
            // Validate that we have processed at least one face successfully
            if (avatarData.FaceRegions.Count == 0)
            {
                throw new InvalidOperationException($"No faces detected in any of the {avatarTextures.Length} avatar textures. Please check that the images contain visible faces.");
            }
            
            if (avatarData.Latents.Count == 0)
            {
                throw new InvalidOperationException($"Failed to generate latents for any of the {avatarData.FaceRegions.Count} detected faces. Check VAE encoder initialization and texture format.");
            }
            
            if (avatarData.Latents.Count != avatarData.FaceRegions.Count)
            {
                Logger.LogWarning($"[MuseTalkInference] Latent count ({avatarData.Latents.Count}) does not match face region count ({avatarData.FaceRegions.Count}). Some faces may have failed processing.");
            }
            
            // Store avatar data for later use in blending
            _avatarData = avatarData;
            
            // Cache the processed avatar animation for reuse
            ManageCacheSize();
            _avatarAnimationCache[memoryCacheKey] = avatarData;
            
            // Save to disk cache for persistent storage
            if (_diskCache != null && !string.IsNullOrEmpty(diskCacheKey))
            {
                try
                {
                    await _diskCache.SaveAvatarDataAsync(diskCacheKey, avatarData);
                    Logger.Log($"[MuseTalkInference] Saved avatar data to disk cache: {diskCacheKey}");
                }
                catch (Exception e)
                {
                    Logger.LogWarning($"[MuseTalkInference] Failed to save to disk cache: {e.Message}");
                }
            }
            
            Logger.Log($"[MuseTalkInference] Successfully processed {avatarData.FaceRegions.Count} face regions with {avatarData.Latents.Count} latent sets across {avatarTextures.Length} avatar textures");
            
            return avatarData;
        }
        
        /// <summary>
        /// Process audio clip and extract Whisper features
        /// </summary>
        private async Task<AudioFeatures> ProcessAudio(AudioClip audioClip)
        {
            if (audioClip == null)
                throw new ArgumentNullException(nameof(audioClip));
                
            // Convert AudioClip to float array
            var audioData = AudioUtils.AudioClipToFloatArray(audioClip);
            
            // Convert stereo to mono if needed
            if (audioClip.channels == 2)
            {
                audioData = AudioUtils.StereoToMono(audioData);
            }
            
            AudioFeatures features;
            
            if (_whisperModel == null || !_whisperModel.IsInitialized)
            {
                throw new InvalidOperationException("Whisper model is not initialized. Real Whisper model is required for proper inference.");
            }
            
            // Use ONNX Whisper model
            features = await ExtractWhisperFeatures(audioData, audioClip.frequency);
            
            return features;
        }
        
        /// <summary>
        /// Extract Whisper features using ONNX WhisperModel
        /// </summary>
        private async Task<AudioFeatures> ExtractWhisperFeatures(float[] audioData, int sampleRate)
        {
            return await Task.Run(() =>
            {
                var features = _whisperModel.ProcessAudio(audioData, sampleRate);
                
                if (features == null)
                {
                    throw new InvalidOperationException("Whisper model failed to process audio. Check model loading and input data.");
                }
                
                return features;
            });
        }
        
        /// <summary>
        /// Encode image using VAE encoder
        /// REFACTORED: Works with byte arrays instead of Texture2D
        /// </summary>
        private async Task<float[]> EncodeImage(byte[] imageData, int width, int height)
        {
            return await Task.Run(() =>
            {
                // Resize image data to 256x256 for VAE encoder
                var resizedData = TextureUtils.ResizeTextureToExactSize(imageData, width, height, 256, 256, TextureUtils.SamplingMode.Bilinear);
                var inputTensor = TextureUtils.BytesToTensor(resizedData, 256, 256);
                
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("image", inputTensor)
                };
                
                // Run VAE encoder
                using var results = _vaeEncoderSession.Run(inputs);

                var latents = results.First(r => r.Name == "latents").AsTensor<float>();
                var result = latents.ToArray();
                
                return result;
            });
        }
        
        /// <summary>
        /// Encode image with lower half masked using VAE encoder
        /// Matches Python's encode_image_with_half_mask exactly
        /// REFACTORED: Works with byte arrays instead of Texture2D
        /// </summary>
        private async Task<float[]> EncodeImageWithMask(byte[] imageData, int width, int height)
        {
            return await Task.Run(() =>
            {
                // Resize image data to 256x256 for VAE encoder
                var resizedData = TextureUtils.ResizeTextureToExactSize(imageData, width, height, 256, 256, TextureUtils.SamplingMode.Bilinear);
                var inputTensor = TextureUtils.BytesToTensorWithMask(resizedData, 256, 256);
                
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("image", inputTensor)
                };
                
                // Run VAE encoder
                using var results = _vaeEncoderSession.Run(inputs);
                var latents = results.First(r => r.Name == "latents").AsTensor<float>();
                
                return latents.ToArray();
            });
        }
        
        /// <summary>
        /// Get latents for UNet inference, matching Python's get_latents_for_unet exactly
        /// REFACTORED: Works with byte arrays instead of Texture2D
        /// </summary>
        private async Task<float[]> GetLatentsForUNet(byte[] imageData, int width, int height)
        {
            // Get masked latents (lower half masked)
            var maskedLatents = await EncodeImageWithMask(imageData, width, height);
            
            // Get reference latents (full image)
            var refLatents = await EncodeImage(imageData, width, height);
            
            // Match Python concatenation exactly along axis=1 (channel dimension)
            if (maskedLatents.Length != refLatents.Length)
            {
                throw new InvalidOperationException("Masked and reference latents must have same size");
            }
            
            const int batch = 1;
            const int maskedChannels = 4;
            const int refChannels = 4;
            const int totalChannels = maskedChannels + refChannels;
            const int latentHeight = 32;
            const int latentWidth = 32;
            const int spatialSize = latentHeight * latentWidth;
            
            var combinedLatents = new float[batch * totalChannels * latentHeight * latentWidth];
            
            unsafe
            {
                fixed (float* maskedPtr = maskedLatents)
                fixed (float* refPtr = refLatents)
                fixed (float* combinedPtr = combinedLatents)
                {
                    // Copy channel by channel using fast memory operations
                    for (int b = 0; b < batch; b++)
                    {
                        // Copy masked latents channels (0-3) using Buffer.MemoryCopy
                        for (int c = 0; c < maskedChannels; c++)
                        {
                            int srcOffset = b * maskedChannels * spatialSize + c * spatialSize;
                            int dstOffset = b * totalChannels * spatialSize + c * spatialSize;
                            
                            Buffer.MemoryCopy(
                                maskedPtr + srcOffset,
                                combinedPtr + dstOffset,
                                spatialSize * sizeof(float),
                                spatialSize * sizeof(float)
                            );
                        }
                        
                        // Copy reference latents channels (4-7) using Buffer.MemoryCopy
                        for (int c = 0; c < refChannels; c++)
                        {
                            int srcOffset = b * refChannels * spatialSize + c * spatialSize;
                            int dstOffset = b * totalChannels * spatialSize + (maskedChannels + c) * spatialSize;
                            
                            Buffer.MemoryCopy(
                                refPtr + srcOffset,
                                combinedPtr + dstOffset,
                                spatialSize * sizeof(float),
                                spatialSize * sizeof(float)
                            );
                        }
                    }
                }
            }
            
            return combinedLatents;
        }
        
        /// <summary>
        /// Generate video frames using UNet and VAE decoder
        /// </summary>
        private async Task<List<Texture2D>> GenerateFrames(AvatarData avatarData, AudioFeatures audioFeatures, int batchSize)
        {
            var frames = new List<Texture2D>();
            
            // Use audio length to determine frame count (like Python implementation)
            int numFrames = audioFeatures.FeatureChunks.Count;
            int numBatches = Mathf.CeilToInt((float)numFrames / batchSize);
            
            if (avatarData.Latents.Count == 0)
            {
                Logger.LogError("[MuseTalkInference] No avatar latents available for frame generation");
                return frames;
            }
            
            Logger.Log($"[MuseTalkInference] Processing {numFrames} frames in {numBatches} batches of {batchSize}");
            
            // Create cycled latent list for smooth animation
            var cycleDLatents = new List<float[]>(avatarData.Latents);
            var reversedLatents = new List<float[]>(avatarData.Latents);
            reversedLatents.Reverse();
            cycleDLatents.AddRange(reversedLatents);
            
            for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
            {   
                int startIdx = batchIdx * batchSize;
                int endIdx = Math.Min(startIdx + batchSize, numFrames);
                int actualBatchSize = endIdx - startIdx;
                
                Logger.Log($"[MuseTalkInference] Processing batch {batchIdx + 1}/{numBatches} (frames {startIdx}-{endIdx - 1})");
                
                // Prepare batch data using the frame index to cycle through latents
                var latentBatch = PrepareLatentBatchWithCycling(cycleDLatents, startIdx, actualBatchSize);
                var audioBatch = PrepareAudioBatch(audioFeatures.FeatureChunks, startIdx, actualBatchSize);
                
                // Add positional encoding to audio
                var audioWithPE = await AddPositionalEncoding(audioBatch);
                
                // Run UNet inference
                var predictedLatents = await RunUNet(latentBatch, audioWithPE);
                
                // Decode latents to images
                var batchFrames = await DecodeLatents(predictedLatents, actualBatchSize, startIdx);
                
                frames.AddRange(batchFrames);
                
                // Log batch performance
                Logger.Log($"[MuseTalkInference] Batch {batchIdx} completed ({actualBatchSize} frames)");
            }
            
            return frames;
        }
        
        /// <summary>
        /// Prepare latent batch with proper cycling for frame-based animation
        /// CRITICAL FIX: Match Python input_latent_list_cycle indexing exactly
        /// </summary>
        private DenseTensor<float> PrepareLatentBatchWithCycling(List<float[]> cycleDLatents, int startIdx, int batchSize)
        {
            // CRITICAL FIX: Match Python tensor preparation exactly
            // Python processes latents as [1, 8, 32, 32] then concatenates for batch
            const int channels = 8, height = 32, width = 32;
            int totalSize = batchSize * channels * height * width;
            var flatBatch = new float[totalSize];
            
            for (int i = 0; i < batchSize; i++)
            {
                // Match Python latent cycling exactly
                int globalFrameIdx = startIdx + i;
                var latentIdx = globalFrameIdx % cycleDLatents.Count;
                var latent = cycleDLatents[latentIdx];
                
                // Verify latent size
                const int expectedLatentSize = 1 * channels * height * width;
                if (latent.Length != expectedLatentSize)
                {
                    throw new InvalidOperationException($"Latent size mismatch: got {latent.Length}, expected {expectedLatentSize}");
                }
                
                // OPTIMIZATION 4: Use unsafe copy for latent batch preparation
                const int singleLatentSize = channels * height * width;
                int batchOffset = i * singleLatentSize;
                
                unsafe
                {
                    fixed (float* src = latent)
                    fixed (float* dst = &flatBatch[batchOffset])
                    {
                        Buffer.MemoryCopy(src, dst, singleLatentSize * sizeof(float), singleLatentSize * sizeof(float));
                    }
                }
            }
            
            // Return tensor with proper batch dimension: [batchSize, 8, 32, 32]
            return new DenseTensor<float>(flatBatch, new[] { batchSize, channels, height, width });
        }
        
        /// <summary>
        /// Prepare audio batch for processing
        /// </summary>
        private DenseTensor<float> PrepareAudioBatch(List<float[]> audioChunks, int startIdx, int batchSize)
        {
            // Audio chunks are [50, 384] and we need [batchSize, 50, 384]
            int timeSteps = 50, features = 384;
            int totalSize = batchSize * timeSteps * features;
            var flatBatch = new float[totalSize];
            
            for (int i = 0; i < batchSize; i++)
            {
                var audioIdx = startIdx + i;
                
                if (audioIdx < audioChunks.Count)
                {
                    var chunk = audioChunks[audioIdx];
                    int batchOffset = i * timeSteps * features;
                    
                    for (int idx = 0; idx < chunk.Length && idx < timeSteps * features; idx++)
                    {
                        flatBatch[batchOffset + idx] = chunk[idx];
                    }
                }
            }
            
            return new DenseTensor<float>(flatBatch, new[] { batchSize, timeSteps, features });
        }
        
        /// <summary>
        /// Add positional encoding to audio features
        /// </summary>
        private async Task<DenseTensor<float>> AddPositionalEncoding(DenseTensor<float> audioBatch)
        {
            return await Task.Run(() =>
            {
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("audio_features", audioBatch)
                };
                
                using var results = _positionalEncodingSession.Run(inputs);
                var output = results.First().AsTensor<float>();
                var result = new DenseTensor<float>(output.ToArray(), output.Dimensions.ToArray());
                
                return result;
            });
        }
        
        /// <summary>
        /// Run UNet inference to predict new latents
        /// CRITICAL FIX: Match Python UNet input processing exactly
        /// OPTIMIZED: Eliminate unnecessary copy by returning ONNX tensor directly with persistent memory
        /// </summary>
        private async Task<Tensor<float>> RunUNet(DenseTensor<float> latentBatch, DenseTensor<float> audioBatch)
        {
            return await Task.Run(() =>
            {
                // Match Python timestep preparation
                int batchSize = (int)latentBatch.Dimensions[0];
                var timesteps = new long[batchSize];
                for (int i = 0; i < batchSize; i++)
                    timesteps[i] = 0L;
                
                var timestepTensor = new DenseTensor<long>(timesteps, new[] { batchSize });
                
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input_latents", latentBatch),
                    NamedOnnxValue.CreateFromTensor("timesteps", timestepTensor),
                    NamedOnnxValue.CreateFromTensor("audio_prompts", audioBatch)
                };
                
                _reusableUNetResult?.Dispose();
                
                // Keep the result alive as a member variable to prevent GC of tensor memory
                _reusableUNetResult = _unetSession.Run(inputs);
                var output = _reusableUNetResult.First().AsTensor<float>();
                
                // OPTIMIZATION: Return the tensor directly without copying via ToArray()
                // The tensor memory is kept alive by _reusableUNetResult member variable
                Tensor<float> result;
                if (output is DenseTensor<float> denseTensor)
                {
                    // Create a new DenseTensor that shares the same Memory<float> buffer (zero copy!)
                    // Memory remains valid because _reusableUNetResult keeps it alive
                    result = new DenseTensor<float>(denseTensor.Buffer, denseTensor.Dimensions.ToArray());
                }
                else
                {
                    // Fallback: only copy if we can't access the buffer directly
                    Logger.Log($"[MuseTalkInference] UNet_ONNX_Inference: Fallback to ToArray()");
                    result = new DenseTensor<float>(output.ToArray(), output.Dimensions.ToArray());
                }
                
                return result;
            });
        }
        
        /// <summary>
        /// Decode latents back to images using VAE decoder and apply seamless blending
        /// OPTIMIZED: Uses reusable batch array and efficient memory operations with zero-copy tensor reuse
        /// </summary>
        private async Task<List<Texture2D>> DecodeLatents(Tensor<float> unetOutputBatch, int batchSize, int globalStartIdx = 0)
        {
            // Run ONNX VAE decoder inference on background thread first            
            var decodedTensors = await Task.Run(() =>
            {
                var tensors = new List<Tensor<float>>();
                Tensor<float> batchImageOutput = null; // Keep reference alive
                
                try
                {
                    // OPTIMIZATION: Reuse the tensor directly without creating a copy
                    // CreateFromTensor with existing tensor should not copy if tensor owns its buffer
                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("latents", unetOutputBatch)
                    };
                    
                    using var results = _vaeDecoderSession.Run(inputs);

                    var batchImageOutputValue = results.First();
                    batchImageOutput = batchImageOutputValue.AsTensor<float>(); // Keep alive
                    
                    // OPTIMIZATION 1: Pre-calculate dimensions once for entire batch
                    var imageDims = new int[batchImageOutput.Dimensions.Length - 1]; // Remove batch dimension
                    for (int i = 1; i < batchImageOutput.Dimensions.Length; i++)
                    {
                        imageDims[i - 1] = (int)batchImageOutput.Dimensions[i];
                    }
                    
                    int imagesPerBatch = imageDims.Aggregate(1, (a, b) => a * b);
                    int totalElements = imagesPerBatch * batchSize;
                    
                    if (_reusableBatchArray.Length < totalElements)
                    {
                        _reusableBatchArray = new float[totalElements];
                    }
                    
                    if (batchImageOutputValue.Value is DenseTensor<float> denseTensor)
                    {
                        // Use DenseTensor.Buffer for direct access
                        denseTensor.Buffer.Span.CopyTo(_reusableBatchArray.AsSpan(0, totalElements));
                    }
                    else
                    {
                        // Fallback to ToArray() if not DenseTensor
                        var sourceArray = batchImageOutput.ToArray();
                        sourceArray.CopyTo(_reusableBatchArray.AsSpan(0, totalElements));
                    }
                    
                    for (int b = 0; b < batchSize; b++)
                    {
                        int offset = b * imagesPerBatch;
                        
                        // Create Memory<float> that references the reusable array directly (no copying!)
                        var imageMemory = _reusableBatchArray.AsMemory(offset, imagesPerBatch);
                        var singleImageTensor = new DenseTensor<float>(imageMemory, imageDims);
                        tensors.Add(singleImageTensor);
                    }
                }
                catch (Exception e)
                {
                    Logger.LogWarning($"[MuseTalkInference] Batch VAE decoding failed: {e.Message}");
                    throw;
                }
                return tensors;
            });
            
            // Convert tensors to textures and apply blending on main thread
            var blendedFrames = new List<Texture2D>();
            
            for (int i = 0; i < decodedTensors.Count; i++)
            {
                var tensor = decodedTensors[i];
                
                // Calculate global frame index for proper numbering
                int globalFrameIdx = globalStartIdx + i;
                
                // Step 1: Convert tensor to raw decoded texture
                var rawDecodedTexture = TextureUtils.TensorToTexture2D(tensor);
                
                // Step 2: Resize to face crop dimensions (matching Python cv2.resize)
                if (_avatarData != null && _avatarData.FaceRegions.Count > 0)
                {
                    // Get corresponding face bbox for sizing
                    int avatarIndex = globalFrameIdx % (2 *_avatarData.FaceRegions.Count); // cycled latents
                    if (avatarIndex >= _avatarData.FaceRegions.Count)
                    {
                        avatarIndex = (2 * _avatarData.FaceRegions.Count) - 1 - avatarIndex;
                    }
                    var faceData = _avatarData.FaceRegions[avatarIndex];
                    var bbox = faceData.BoundingBox;
                    
                    int targetWidth = Mathf.RoundToInt(bbox.width);
                    int targetHeight = Mathf.RoundToInt(bbox.height);
                    
                    if (_config.Version == "v15")
                    {
                        targetHeight += 10; // extra_margin
                        // Clamp to original image height
                        if (faceData.OriginalTextureData != null)
                        {
                            targetHeight = Mathf.Min(targetHeight, faceData.OriginalHeight - Mathf.RoundToInt(bbox.y));
                        }
                    }
                    
                    // Resize decoded frame to face crop size
                    var resizedFrame = TextureUtils.ResizeTextureToExactSize(rawDecodedTexture, targetWidth, targetHeight);
                    
                    // Step 3: Apply seamless face blending
                    if (faceData.OriginalTextureData == null)
                    {
                        blendedFrames.Add(resizedFrame);
                        continue;
                    }
                    
                    // Convert byte arrays back to textures for blending (only when needed)
                    var originalImage = TextureUtils.BytesToTexture2D(faceData.OriginalTextureData, faceData.OriginalWidth, faceData.OriginalHeight);
                    var blurredMask = TextureUtils.BytesToTexture2D(faceData.BlurredMaskData, faceData.BlurredMaskWidth, faceData.BlurredMaskHeight);
                    var faceLarge = TextureUtils.BytesToTexture2D(faceData.FaceLargeData, faceData.FaceLargeWidth, faceData.FaceLargeHeight);
                    
                    // Convert face bbox to Vector4 format for blending (x1, y1, x2, y2)
                    var faceBbox = new Vector4(bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height);
                    string blendingMode = "jaw"; // version v15
                    
                    try
                    {
                        // Use precomputed segmentation data for optimal performance
                        var blendedFrame = ImageBlendingHelper.BlendFaceWithOriginal(
                            originalImage,
                            resizedFrame,
                            faceBbox,
                            blendingMode,
                            faceData.CropBox, // Use cached crop box
                            blurredMask, // Use precomputed blurred mask
                            faceLarge); // Use precomputed face large
                        
                        // Clean up temporary textures
                        UnityEngine.Object.Destroy(originalImage);
                        UnityEngine.Object.Destroy(blurredMask);
                        UnityEngine.Object.Destroy(faceLarge);
                        UnityEngine.Object.Destroy(resizedFrame);
                        
                        blendedFrames.Add(blendedFrame);
                    }
                    catch (Exception e)
                    {
                        Logger.LogError($"[MuseTalkInference] Frame processing failed for frame {globalFrameIdx}: {e.Message}");
                        
                        // Clean up temporary textures in case of error
                        UnityEngine.Object.Destroy(originalImage);
                        UnityEngine.Object.Destroy(blurredMask);
                        UnityEngine.Object.Destroy(faceLarge);
                        
                        blendedFrames.Add(resizedFrame);
                    }
                }
                else
                {
                    blendedFrames.Add(rawDecodedTexture);
                }    
            }
            
            return blendedFrames;
        }
        
        /// <summary>
        /// Clear the avatar animation cache (useful for memory management)
        /// REFACTORED: No texture cleanup needed since we use byte arrays now
        /// </summary>
        public static void ClearAvatarAnimationCache()
        {
            // No need to destroy textures since we use byte arrays now
            // The garbage collector will handle the byte arrays automatically
            _avatarAnimationCache.Clear();
            Logger.Log("[MuseTalkInference] Cleared avatar animation cache");
        }
        
        /// <summary>
        /// Clear the disk cache (useful for troubleshooting or storage management)
        /// </summary>
        public async Task ClearDiskCacheAsync()
        {
            if (_diskCache != null)
            {
                await _diskCache.ClearCache();
                Logger.Log("[MuseTalkInference] Cleared disk cache");
            }
        }
        
        /// <summary>
        /// Get disk cache statistics (if disk cache is enabled)
        /// </summary>
        public CacheStatistics GetDiskCacheStatistics()
        {
            return _diskCache?.GetCacheStatistics();
        }
        
        /// <summary>
        /// Check if disk cache is enabled and functional
        /// </summary>
        public bool IsDiskCacheEnabled => _config.EnableDiskCache && _diskCache != null;
        
        /// <summary>
        /// Get cache information for debugging
        /// </summary>
        public string GetCacheInfo()
        {
            var memoryEntries = _avatarAnimationCache.Count;
            var diskStats = GetDiskCacheStatistics();
            
            if (diskStats != null)
            {
                return $"Memory Cache: {memoryEntries} entries | Disk Cache: {diskStats.TotalEntries} entries, {diskStats.TotalSizeMB:F1}MB, {diskStats.HitRate:P1} hit rate";
            }
            else
            {
                return $"Memory Cache: {memoryEntries} entries | Disk Cache: Disabled";
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _unetSession?.Dispose();
                _vaeEncoderSession?.Dispose();
                _vaeDecoderSession?.Dispose();
                _positionalEncodingSession?.Dispose();
                _whisperModel?.Dispose();
                _faceAnalysis?.Dispose();
                _reusableUNetResult?.Dispose();
                _diskCache?.Dispose();
                
                _disposed = true;
                Logger.Log("[MuseTalkInference] Disposed");
            }
            
            GC.SuppressFinalize(this);
        }
        
        ~MuseTalkInference()
        {
            Dispose();
        }
    }
} 