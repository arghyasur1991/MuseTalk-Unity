using System;
using System.IO;
using System.Linq;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Unity.Collections.LowLevel.Unsafe;

namespace MuseTalk.Utils
{
    using Models;
    /// <summary>
    /// ONNX-based Face Parsing using BiSeNet model for accurate face segmentation
    /// Replaces geometric approximations with neural network-based face parsing
    /// </summary>
    public class FaceParsingHelper : IDisposable
    {
        private static readonly DebugLogger Logger = new();
        
        private InferenceSession _session;
        private bool _initialized = false;
        private bool _disposed = false;
        
        // BiSeNet face parsing classes (19 total)
        public enum FaceParsingClass
        {
            Background = 0,
            Skin = 1,           // Face region
            LeftBrow = 2,
            RightBrow = 3,
            LeftEye = 4,
            RightEye = 5,
            EyeGlass = 6,
            LeftEar = 7,
            RightEar = 8,
            Earring = 9,
            Nose = 10,
            Mouth = 11,         // Lips region
            UpperLip = 12,
            LowerLip = 13,
            Neck = 14,
            Necklace = 15,
            Cloth = 16,
            Hair = 17,
            Hat = 18
        }
        
        public bool IsInitialized => _initialized;
        
        public FaceParsingHelper(MuseTalkConfig config)
        {
            try
            {
                _session = ModelUtils.LoadModel(config, "face_parsing");
                _initialized = true;
                Logger.Log("[FaceParsingHelper] Successfully initialized");
            }
            catch (Exception e)
            {
                Logger.LogError($"[FaceParsingHelper] Failed to initialize: {e.Message}");
                _initialized = false;
            }
        }
        
        /// <summary>
        /// Generate face segmentation mask using ONNX BiSeNet model from byte array
        /// OPTIMIZED: Works with byte arrays throughout the entire pipeline
        /// </summary>
        public (byte[], int, int) GenerateFaceSegmentationMask(byte[] inputImageData, int width, int height, string mode = "jaw")
        {
            if (!_initialized)
            {
                Logger.LogError("[FaceParsingHelper] Model not initialized");
                return (null, 0, 0);
            }
            
            try
            {
                // Step 1: Preprocess image for BiSeNet (512x512, normalized) directly from byte array
                var preprocessedTensor = PreprocessImageForBiSeNet(inputImageData, width, height);
                
                // Step 2: Run ONNX inference
                var parsingResult = RunBiSeNetInference(preprocessedTensor);
                
                // Step 3: Post-process to create mask based on mode, returning byte array
                var (maskData, maskWidth, maskHeight) = PostProcessParsingResult(parsingResult, mode, width, height);
                
                return (maskData, maskWidth, maskHeight);
            }
            catch (Exception e)
            {
                Logger.LogError($"[FaceParsingHelper] Face parsing failed: {e.Message}");
                return (null, 0, 0);
            }
        }
        
        private DenseTensor<float> PreprocessImageForBiSeNet(byte[] inputImageData, int width, int height)
        {
            // Resize to BiSeNet input size (512x512) - now uses optimized ResizeTextureToExactSize with byte arrays
            var resizedImageData = TextureUtils.ResizeTextureToExactSize(inputImageData, width, height, 512, 512, TextureUtils.SamplingMode.Bilinear);
            
            // ImageNet normalization: (pixel/255 - mean) / std for each channel
            // R: (pixel/255 - 0.485) / 0.229, G: (pixel/255 - 0.456) / 0.224, B: (pixel/255 - 0.406) / 0.225
            // Transform to: pixel * (1/255/std) + (-mean/std)
            var multipliers = new float[] 
            { 
                1.0f / (255.0f * 0.229f),  // R: 1/(255*0.229) 
                1.0f / (255.0f * 0.224f),  // G: 1/(255*0.224)
                1.0f / (255.0f * 0.225f)   // B: 1/(255*0.225)
            };
            var offsets = new float[] 
            { 
                -0.485f / 0.229f,  // R: -mean_r/std_r
                -0.456f / 0.224f,  // G: -mean_g/std_g  
                -0.406f / 0.225f   // B: -mean_b/std_b
            };
            
            return TextureUtils.PreprocessImageOptimized(resizedImageData, 512, 512, multipliers, offsets);
        }
        
        private unsafe int[,] RunBiSeNetInference(DenseTensor<float> inputTensor)
        {
            // Create input for ONNX model
            var inputs = new[]
            {
                NamedOnnxValue.CreateFromTensor("image", inputTensor)
            };
            
            // Run inference
            using var results = _session.Run(inputs);
            var output = results.First(r => r.Name == "face_parsing_output").AsTensor<float>();
            
            // Convert output to segmentation map [512, 512]
            // Output shape: [1, 19, 512, 512] - 19 face parsing classes
            var parsingMap = new int[512, 512];
            
            // Get tensor data array - unfortunately ONNX tensors don't expose direct memory access
            var outputArray = output.ToArray();
            
            // Pre-calculate tensor strides for efficient pointer arithmetic
            // Tensor layout: [batch=1, classes=19, height=512, width=512]
            const int imageSize = 512 * 512;
            const int classStride = imageSize; // Elements per class channel
            
            // OPTIMIZED: Get unsafe pointer to array data outside parallel loop
            fixed (float* outputPtr = outputArray)
            {
                // Convert pointer to IntPtr to pass into parallel lambda (C# limitation workaround)
                IntPtr outputPtrAddr = new(outputPtr);
                
                // OPTIMIZED: Maximum parallelism across all 512×512 pixels (262,144-way parallelism)
                // Apply argmax operation with direct unsafe memory access and stride-based calculation
                System.Threading.Tasks.Parallel.For(0, imageSize, pixelIndex =>
                {
                    // Convert IntPtr back to unsafe pointer inside lambda
                    float* unsafeOutputPtr = (float*)outputPtrAddr.ToPointer();
                    
                    // Calculate x, y coordinates from linear pixel index using bit operations
                    int y = pixelIndex >> 9;  // Divide by 512 (right shift 9 bits: 2^9 = 512)
                    int x = pixelIndex & 511; // Modulo 512 (bitwise AND with 511: 2^9-1 = 511)
                    
                    // Find class with maximum probability using direct unsafe memory access
                    int maxClass = 0;
                    float* pixelPtr = unsafeOutputPtr + pixelIndex; // Pointer to class 0 for this pixel
                    float maxProb = *pixelPtr; // Dereference pointer - no bounds checking!
                    
                    // OPTIMIZED: Direct pointer arithmetic for argmax (19 classes total)
                    // Check classes 1-18 using pointer arithmetic - fastest possible access
                    for (int c = 1; c < 19; c++)
                    {
                        float* classPtr = pixelPtr + c * classStride; // Pointer to class c for this pixel
                        float prob = *classPtr; // Direct memory access - no bounds checking!
                        if (prob > maxProb)
                        {
                            maxProb = prob;
                            maxClass = c;
                        }
                    }
                    
                    // Store result in parsing map
                    parsingMap[y, x] = maxClass;
                });
            }
            
            return parsingMap;
        }
        
        private unsafe (byte[], int, int) PostProcessParsingResult(int[,] parsingMap, string mode, int targetWidth, int targetHeight)
        {
            // Create mask data directly as byte array (RGB24: 3 bytes per pixel)
            const int maskWidth = 512;
            const int maskHeight = 512;
            int totalBytes = maskWidth * maskHeight * 3;
            var maskData = new byte[totalBytes];
            
            // Pre-calculate class IDs for each mode to avoid string comparison in hot path
            bool* classLookup = stackalloc bool[19]; // 19 face parsing classes
            
            string lowerMode = mode.ToLower();
            if (lowerMode == "neck")
            {
                // Include face, lips, and neck regions
                classLookup[(int)FaceParsingClass.Skin] = true;
                classLookup[(int)FaceParsingClass.Mouth] = true;
                classLookup[(int)FaceParsingClass.UpperLip] = true;
                classLookup[(int)FaceParsingClass.LowerLip] = true;
                classLookup[(int)FaceParsingClass.Neck] = true;
            }
            else if (lowerMode == "jaw")
            {
                // Include face and mouth regions (for talking head)
                classLookup[(int)FaceParsingClass.Skin] = true;
                classLookup[(int)FaceParsingClass.Mouth] = true;
                classLookup[(int)FaceParsingClass.UpperLip] = true;
                classLookup[(int)FaceParsingClass.LowerLip] = true;
            }
            else // "raw" or default
            {
                // Include face and lip regions
                classLookup[(int)FaceParsingClass.Skin] = true;
                classLookup[(int)FaceParsingClass.Mouth] = true;
                classLookup[(int)FaceParsingClass.UpperLip] = true;
                classLookup[(int)FaceParsingClass.LowerLip] = true;
            }
            
            // OPTIMIZED: Maximum parallelism across all 512×512 pixels (262,144-way parallelism)
            // Apply mode-specific processing with stride-based coordinate calculation
            const int imageSize = 512 * 512;
            
            fixed (byte* maskPtrFixed = maskData)
            {
                // Capture pointer in local variable to avoid lambda closure issues
                byte* maskPtrLocal = maskPtrFixed;
                
                System.Threading.Tasks.Parallel.For(0, imageSize, pixelIndex =>
                {
                    // Calculate x, y coordinates from linear pixel index using bit operations
                    int y = pixelIndex >> 9;  // Divide by 512 (right shift 9 bits: 2^9 = 512)
                    int x = pixelIndex & 511; // Modulo 512 (bitwise AND with 511: 2^9-1 = 511)
                    
                    // Get class ID from parsing map
                    int classId = parsingMap[y, x];
                    
                    // Fast lookup using pre-calculated boolean array (no switch statement)
                    bool isForeground = classId < 19 && classLookup[classId];
                    
                    // Calculate target pixel pointer using stride arithmetic (no Y-flipping needed for byte arrays)
                    byte* pixelPtr = maskPtrLocal + ((y << 9) + x) * 3; // (y * 512 + x) * 3 for RGB24
                    
                    // Set mask value directly in memory (RGB24: all channels same for grayscale)
                    byte maskValue = isForeground ? (byte)255 : (byte)0;
                    pixelPtr[0] = maskValue; // R
                    pixelPtr[1] = maskValue; // G  
                    pixelPtr[2] = maskValue; // B
                });
            }
            
            // Resize to target dimensions if needed using optimized resize
            if (targetWidth != 512 || targetHeight != 512)
            {
                var resizedMaskData = TextureUtils.ResizeTextureToExactSize(maskData, maskWidth, maskHeight, targetWidth, targetHeight, TextureUtils.SamplingMode.Bilinear);
                return (resizedMaskData, targetWidth, targetHeight);
            }
            
            return (maskData, maskWidth, maskHeight);
        }
        
        /// <summary>
        /// Create face mask with morphological operations returning byte array (matching Python implementation)
        /// </summary>
        public (byte[], int, int) CreateFaceMaskWithMorphology(byte[] inputImage, int width, int height, string mode = "jaw")
        {
            var (baseMaskData, baseMaskWidth, baseMaskHeight) = GenerateFaceSegmentationMask(inputImage, width, height, mode);
            if (baseMaskData == null) return (null, 0, 0);
            
            var (smoothedMaskData, smoothedMaskWidth, smoothedMaskHeight) = ApplyMorphologicalOperations(baseMaskData, baseMaskWidth, baseMaskHeight, mode);
            
            return (smoothedMaskData, smoothedMaskWidth, smoothedMaskHeight);
        }
        
        private unsafe (byte[], int, int) ApplyMorphologicalOperations(byte[] maskData, int width, int height, string mode)
        {
            int totalPixels = width * height;
            int totalBytes = totalPixels * 3; // RGB24: 3 bytes per pixel
            
            // Allocate working buffers
            byte* dilatedPtr = (byte*)UnsafeUtility.Malloc(totalBytes, 4, Unity.Collections.Allocator.Temp);
            byte* erodedPtr = (byte*)UnsafeUtility.Malloc(totalBytes, 4, Unity.Collections.Allocator.Temp);
            byte* blurredPtr = (byte*)UnsafeUtility.Malloc(totalBytes, 4, Unity.Collections.Allocator.Temp);
            
            try
            {
                fixed (byte* inputPtr = maskData)
                {
                    // Apply morphological operations using unsafe pointers
                    TextureUtils.ApplyDilationUnsafe(inputPtr, dilatedPtr, width, height, 3);
                    TextureUtils.ApplyErosionUnsafe(dilatedPtr, erodedPtr, width, height, 2);
                    
                    // Apply optimized Gaussian blur
                    float sigma = 1.0f;
                    int kernelSize = Mathf.RoundToInt(sigma * 6) | 1; // Ensure odd size
                    kernelSize = Mathf.Max(kernelSize, 3); // Minimum kernel size
                    
                    TextureUtils.ApplySimpleGaussianBlur(erodedPtr, blurredPtr, width, height, kernelSize);
                    
                    // Create result byte array and copy data
                    var result = new byte[totalBytes];
                    fixed (byte* resultPtr = result)
                    {
                        UnsafeUtility.MemCpy(resultPtr, blurredPtr, totalBytes);
                    }
                    
                    return (result, width, height);
                }
            }
            finally
            {
                // Clean up temporary buffers
                UnsafeUtility.Free(dilatedPtr, Unity.Collections.Allocator.Temp);
                UnsafeUtility.Free(erodedPtr, Unity.Collections.Allocator.Temp);
                UnsafeUtility.Free(blurredPtr, Unity.Collections.Allocator.Temp);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _disposed = true;
                Logger.Log("[FaceParsingHelper] Disposed");
            }
        }
        
        ~FaceParsingHelper()
        {
            Dispose();
        }
    }
} 