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
        /// Generate face segmentation mask using ONNX BiSeNet model
        /// </summary>
        public Texture2D GenerateFaceSegmentationMask(Texture2D inputImage, string mode = "jaw")
        {
            if (!_initialized)
            {
                Logger.LogError("[FaceParsingHelper] Model not initialized");
                return null;
            }
            
            try
            {
                // Step 1: Preprocess image for BiSeNet (512x512, normalized)
                var preprocessedTensor = PreprocessImageForBiSeNet(inputImage);
                
                // Step 2: Run ONNX inference
                var parsingResult = RunBiSeNetInference(preprocessedTensor);
                
                // Step 3: Post-process to create mask based on mode
                var mask = PostProcessParsingResult(parsingResult, mode, inputImage.width, inputImage.height);
                
                return mask;
            }
            catch (Exception e)
            {
                Logger.LogError($"[FaceParsingHelper] Face parsing failed: {e.Message}");
                return null;
            }
        }
        
        private unsafe DenseTensor<float> PreprocessImageForBiSeNet(Texture2D inputImage)
        {
            // Resize to BiSeNet input size (512x512) - now uses optimized ResizeTextureToExactSize
            var resizedImage = TextureUtils.ResizeTextureToExactSize(inputImage, 512, 512);
            
            // Create tensor data array - [batch, channels, height, width] = [1, 3, 512, 512]
            var tensorData = new float[1 * 3 * 512 * 512];
            
            // ImageNet normalization values used by BiSeNet (pre-calculated for performance)
            const float meanR = 0.485f, meanG = 0.456f, meanB = 0.406f;
            const float stdR = 0.229f, stdG = 0.224f, stdB = 0.225f;
            const float invStdR = 1.0f / stdR, invStdG = 1.0f / stdG, invStdB = 1.0f / stdB;
            
            // Get direct access to resized image pixel data
            var pixelData = resizedImage.GetPixelData<byte>(0);
            byte* pixelPtr = (byte*)pixelData.GetUnsafeReadOnlyPtr();
            
            // Pre-calculate tensor offsets for each channel (CHW format)
            const int imageSize = 512 * 512;
            
            // OPTIMIZED: Maximum parallelism across all 512×512 pixels (262,144-way parallelism)
            // Process in CHW format (channels first) with stride-based coordinate calculation
            System.Threading.Tasks.Parallel.For(0, imageSize, pixelIndex =>
            {
                // Calculate x, y coordinates from linear pixel index using stride arithmetic
                int y = pixelIndex >> 9;  // Divide by 512 (right shift 9 bits: 2^9 = 512) - faster than division
                int x = pixelIndex & 511; // Modulo 512 (bitwise AND with 511: 2^9-1 = 511) - faster than modulo
                
                // CRITICAL: Unity GetPixelData() is bottom-left origin, flip Y for ONNX (top-left)
                int unityY = 511 - y; // Flip Y coordinate for ONNX coordinate system
                
                // Calculate pointer for this specific pixel using stride arithmetic
                byte* pixelBytePtr = pixelPtr + ((unityY << 9) + x) * 3; // unityY * 512 + x, then * 3 for RGB24
                
                // Process all 3 channels for this pixel with unrolled loop for maximum performance
                {
                    // R channel (channel 0)
                    float normalizedR = (pixelBytePtr[0] / 255.0f - meanR) * invStdR;
                    tensorData[y * 512 + x] = normalizedR; // Channel 0 offset: 0 * imageSize
                    
                    // G channel (channel 1)  
                    float normalizedG = (pixelBytePtr[1] / 255.0f - meanG) * invStdG;
                    tensorData[imageSize + y * 512 + x] = normalizedG; // Channel 1 offset: 1 * imageSize
                    
                    // B channel (channel 2)
                    float normalizedB = (pixelBytePtr[2] / 255.0f - meanB) * invStdB;
                    tensorData[2 * imageSize + y * 512 + x] = normalizedB; // Channel 2 offset: 2 * imageSize
                }
            });
            UnityEngine.Object.Destroy(resizedImage);
            return new DenseTensor<float>(tensorData, new[] { 1, 3, 512, 512 });
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
        
        private unsafe Texture2D PostProcessParsingResult(int[,] parsingMap, string mode, int targetWidth, int targetHeight)
        {
            // Create mask texture with RGB24 format for maximum efficiency
            var maskTexture = new Texture2D(512, 512, TextureFormat.RGB24, false);
            var maskPixelData = maskTexture.GetPixelData<byte>(0);
            
            // Get unsafe pointer for direct memory operations
            byte* maskPtr = (byte*)maskPixelData.GetUnsafePtr();
            
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
            System.Threading.Tasks.Parallel.For(0, imageSize, pixelIndex =>
            {
                // Calculate x, y coordinates from linear pixel index using bit operations
                int y = pixelIndex >> 9;  // Divide by 512 (right shift 9 bits: 2^9 = 512)
                int x = pixelIndex & 511; // Modulo 512 (bitwise AND with 511: 2^9-1 = 511)
                
                // Get class ID from parsing map
                int classId = parsingMap[y, x];
                
                // Fast lookup using pre-calculated boolean array (no switch statement)
                bool isForeground = classId < 19 && classLookup[classId];
                
                // CRITICAL: Flip Y coordinate for Unity's bottom-left origin texture
                int unityY = 511 - y; // Convert from top-left to bottom-left
                
                // Calculate target pixel pointer using stride arithmetic
                byte* pixelPtr = maskPtr + ((unityY << 9) + x) * 3; // (unityY * 512 + x) * 3 for RGB24
                
                // Set mask value directly in memory (RGB24: all channels same for grayscale)
                byte maskValue = isForeground ? (byte)255 : (byte)0;
                pixelPtr[0] = maskValue; // R
                pixelPtr[1] = maskValue; // G  
                pixelPtr[2] = maskValue; // B
            });
            
            // Apply changes to texture (no need for SetPixels since we wrote directly to pixel data)
            maskTexture.Apply();
            
            // Resize to target dimensions if needed using optimized resize
            if (targetWidth != 512 || targetHeight != 512)
            {
                var resizedMask = TextureUtils.ResizeTextureToExactSize(maskTexture, targetWidth, targetHeight);
                UnityEngine.Object.Destroy(maskTexture);
                return resizedMask;
            }
            
            return maskTexture;
        }
        
        /// <summary>
        /// Create face mask with morphological operations (matching Python implementation)
        /// </summary>
        public Texture2D CreateFaceMaskWithMorphology(Texture2D inputImage, string mode = "jaw")
        {
            var baseMask = GenerateFaceSegmentationMask(inputImage, mode);
            if (baseMask == null) return null;
            var smoothedMask = ApplyMorphologicalOperations(baseMask, mode);
            return smoothedMask;
        }
        
        private unsafe Texture2D ApplyMorphologicalOperations(Texture2D mask, string mode)
        {
            int width = mask.width;
            int height = mask.height;
            int totalPixels = width * height;
            int totalBytes = totalPixels * 3; // RGB24: 3 bytes per pixel
            
            // Get input pixel data directly as bytes
            var inputPixelData = mask.GetPixelData<byte>(0);
            byte* inputPtr = (byte*)inputPixelData.GetUnsafeReadOnlyPtr();
            
            // Allocate working buffers
            byte* dilatedPtr = (byte*)UnsafeUtility.Malloc(totalBytes, 4, Unity.Collections.Allocator.Temp);
            byte* erodedPtr = (byte*)UnsafeUtility.Malloc(totalBytes, 4, Unity.Collections.Allocator.Temp);
            byte* blurredPtr = (byte*)UnsafeUtility.Malloc(totalBytes, 4, Unity.Collections.Allocator.Temp);
            
            try
            {
                // Apply morphological operations using unsafe pointers
                ApplyDilationUnsafe(inputPtr, dilatedPtr, width, height, 3);
                ApplyErosionUnsafe(dilatedPtr, erodedPtr, width, height, 2);
                
                // Apply optimized Gaussian blur
                float sigma = 1.0f;
                int kernelSize = Mathf.RoundToInt(sigma * 6) | 1; // Ensure odd size
                kernelSize = Mathf.Max(kernelSize, 3); // Minimum kernel size
                
                TextureUtils.ApplySimpleGaussianBlur(erodedPtr, blurredPtr, width, height, kernelSize);
                
                // Create result texture and copy data
                var result = new Texture2D(width, height, TextureFormat.RGB24, false);
                var resultPixelData = result.GetPixelData<byte>(0);
                byte* resultPtr = (byte*)resultPixelData.GetUnsafePtr();
                
                // Copy final result
                UnsafeUtility.MemCpy(resultPtr, blurredPtr, totalBytes);
                
                result.Apply();
                return result;
            }
            finally
            {
                // Clean up temporary buffers
                UnsafeUtility.Free(dilatedPtr, Unity.Collections.Allocator.Temp);
                UnsafeUtility.Free(erodedPtr, Unity.Collections.Allocator.Temp);
                UnsafeUtility.Free(blurredPtr, Unity.Collections.Allocator.Temp);
            }
        }
        
        /// <summary>
        /// Unsafe optimized dilation operation using direct byte pointer access
        /// OPTIMIZED: Uses parallelization and direct memory operations for maximum performance
        /// </summary>
        private static unsafe void ApplyDilationUnsafe(byte* inputPtr, byte* outputPtr, int width, int height, int kernelSize)
        {
            int radius = kernelSize / 2;
            
            // Parallel processing for maximum performance
            System.Threading.Tasks.Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    byte maxValue = 0;
                    
                    // Check kernel neighborhood
                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            int ny = Mathf.Clamp(y + ky, 0, height - 1);
                            int nx = Mathf.Clamp(x + kx, 0, width - 1);
                            
                            // Get pixel pointer and use red channel as grayscale value
                            byte* samplePixel = inputPtr + (ny * width + nx) * 3;
                            maxValue = (byte)Mathf.Max(maxValue, samplePixel[0]); // Use red channel
                        }
                    }
                    
                    // Set output pixel (RGB24: all channels same for grayscale)
                    byte* outputPixel = outputPtr + (y * width + x) * 3;
                    outputPixel[0] = maxValue; // R
                    outputPixel[1] = maxValue; // G
                    outputPixel[2] = maxValue; // B
                }
            });
        }
        
        /// <summary>
        /// Unsafe optimized erosion operation using direct byte pointer access
        /// OPTIMIZED: Uses parallelization and direct memory operations for maximum performance
        /// </summary>
        private static unsafe void ApplyErosionUnsafe(byte* inputPtr, byte* outputPtr, int width, int height, int kernelSize)
        {
            int radius = kernelSize / 2;
            
            // Parallel processing for maximum performance
            System.Threading.Tasks.Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    byte minValue = 255;
                    
                    // Check kernel neighborhood
                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            int ny = Mathf.Clamp(y + ky, 0, height - 1);
                            int nx = Mathf.Clamp(x + kx, 0, width - 1);
                            
                            // Get pixel pointer and use red channel as grayscale value
                            byte* samplePixel = inputPtr + (ny * width + nx) * 3;
                            minValue = (byte)Mathf.Min(minValue, samplePixel[0]); // Use red channel
                        }
                    }
                    
                    // Set output pixel (RGB24: all channels same for grayscale)
                    byte* outputPixel = outputPtr + (y * width + x) * 3;
                    outputPixel[0] = minValue; // R
                    outputPixel[1] = minValue; // G
                    outputPixel[2] = minValue; // B
                }
            });
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