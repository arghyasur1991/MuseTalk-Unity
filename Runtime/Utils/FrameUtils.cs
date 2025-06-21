using System;
using System.Linq;
using UnityEngine;
using Microsoft.ML.OnnxRuntime.Tensors;
using Unity.Collections.LowLevel.Unsafe;

namespace LiveTalk.Utils
{
    using Core;
    /// <summary>
    /// Utility functions for frame operations
    /// </summary>
    internal static class FrameUtils
    {

#region Conversion

        /// <summary>
        /// Convert a tensor to a frame
        /// </summary>
        /// <param name="tensor">Tensor to convert</param>
        /// <returns>Frame</returns>
        public static unsafe Frame TensorToFrame(Tensor<float> tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
                
            // Assume tensor format is [1, 3, H, W] or [3, H, W]
            var dimensions = tensor.Dimensions.ToArray();
            int height, width, channels;
            int batchOffset = 0;
            
            if (dimensions.Length == 4)
            {
                // Format: [batch, channels, height, width]
                channels = dimensions[1];
                height = dimensions[2];
                width = dimensions[3];
                batchOffset = 0; // Use first batch item
            }
            else if (dimensions.Length == 3)
            {
                // Format: [channels, height, width]
                channels = dimensions[0];
                height = dimensions[1];
                width = dimensions[2];
            }
            else
            {
                throw new ArgumentException($"Unsupported tensor dimensions: {dimensions.Length}");
            }
            
            if (channels != 3)
                throw new ArgumentException($"Expected 3 channels (RGB), got {channels}");

            var pixelData = new byte[width * height * 3];
            
            // Get tensor data array once for performance
            // TODO remove toArray()
            var tensorArray = tensor.ToArray();
            fixed (byte* pixelPtr = pixelData)
            {
                byte* pixelPtrLocal = pixelPtr;
                int channelStride = height * width;
                int batchStride = channels * channelStride;
                
                System.Threading.Tasks.Parallel.For(0, height, y =>
                {
                    byte* rowPtr = pixelPtrLocal + y * width * 3; // RGB24: 3 bytes per pixel
                    for (int x = 0; x < width; x++)
                    {
                        float r, g, b;
                        
                        if (dimensions.Length == 4)
                        {
                            // [batch, channel, height, width] - pre-calculated indices for performance
                            int baseIndex = batchOffset * batchStride + y * width + x;
                            r = tensorArray[baseIndex];                    // Channel 0 (R)
                            g = tensorArray[baseIndex + channelStride];    // Channel 1 (G)
                            b = tensorArray[baseIndex + 2 * channelStride]; // Channel 2 (B)
                        }
                        else
                        {
                            // [channel, height, width] - pre-calculated indices for performance
                            int baseIndex = y * width + x;
                            r = tensorArray[baseIndex];                    // Channel 0 (R)
                            g = tensorArray[baseIndex + channelStride];    // Channel 1 (G)
                            b = tensorArray[baseIndex + 2 * channelStride]; // Channel 2 (B)
                        }

                        r = Mathf.Clamp01(r * 0.5f + 0.5f);
                        g = Mathf.Clamp01(g * 0.5f + 0.5f);
                        b = Mathf.Clamp01(b * 0.5f + 0.5f);
                        
                        // Convert to byte range [0, 255] with proper rounding
                        byte rByte = (byte)Mathf.RoundToInt(r * 255f);
                        byte gByte = (byte)Mathf.RoundToInt(g * 255f);
                        byte bByte = (byte)Mathf.RoundToInt(b * 255f);
                        
                        byte* pixelPtr = rowPtr + x * 3;
                        pixelPtr[0] = rByte; // R
                        pixelPtr[1] = gByte; // G
                        pixelPtr[2] = bByte; // B
                    }
                });
            }
            
            return new Frame(pixelData, width, height);
        }


        /// <summary>
        /// Convert a frame to a tensor
        /// </summary>
        /// <param name="frame">Frame to convert</param>
        /// <param name="width">Image width</param>
        /// <param name="height">Image height</param>
        /// <param name="multiplier">Multiplier for all RGB channels</param>
        /// <param name="offset">Offset for all RGB channels</param>
        /// <param name="applyLowerHalfMask">If true, masks lower half of image to zero (upper half = 1.0, lower half = 0.0)</param>
        public static unsafe DenseTensor<float> FrameToTensor(
            Frame frame, float multiplier, float offset, bool applyLowerHalfMask = false)
        {
            // Use per-channel version with same multiplier/offset for all channels
            var multipliers = new float[] { multiplier, multiplier, multiplier };
            var offsets = new float[] { offset, offset, offset };
            return FrameToTensor(frame, multipliers, offsets, applyLowerHalfMask);
        }

        /// <summary>
        /// Convert a frame to a tensor
        /// </summary>
        /// <param name="frame">Frame to convert</param>
        /// <param name="width">Image width</param>
        /// <param name="height">Image height</param>
        /// <param name="multipliers">Per-channel multipliers (3 values for RGB)</param>
        /// <param name="offsets">Per-channel offsets (3 values for RGB)</param>
        /// <param name="applyLowerHalfMask">If true, masks lower half of image to zero (upper half = 1.0, lower half = 0.0)</param>
        public static unsafe DenseTensor<float> FrameToTensor(
            Frame frame, float[] multipliers, float[] offsets, bool applyLowerHalfMask = false)
        {
            if (multipliers.Length != 3 || offsets.Length != 3)
                throw new ArgumentException("Multipliers and offsets must have exactly 3 elements for RGB channels");
                
            var tensorData = new float[1 * 3 * frame.height * frame.width];
            int imageSize = frame.height * frame.width;
            int halfHeight = applyLowerHalfMask ? frame.height / 2 : 0; // Pre-calculate for mask comparison
            
            // OPTIMIZED: Use unsafe pointers for direct memory access
            fixed (byte* imgPtrFixed = frame.data)
            fixed (float* tensorPtrFixed = tensorData)
            fixed (float* multipliersPtr = multipliers)
            fixed (float* offsetsPtr = offsets)
            {
                // Capture pointers in local variables to avoid lambda closure issues
                byte* imgPtrLocal = imgPtrFixed;
                float* tensorPtrLocal = tensorPtrFixed;
                float* multipliersLocal = multipliersPtr;
                float* offsetsLocal = offsetsPtr;
                
                // MAXIMUM PERFORMANCE: Parallel processing across all pixels for maximum parallelism
                // Process each pixel independently across all available CPU cores
                System.Threading.Tasks.Parallel.For(0, imageSize, pixelIdx =>
                {
                    // Calculate x, y coordinates from linear pixel index for masking
                    int y = pixelIdx / frame.width;
                    
                    // Apply mask if requested (upper half = 1, lower half = 0)
                    float mask = (applyLowerHalfMask && y >= halfHeight) ? 0.0f : 1.0f;
                    
                    // Process all 3 RGB channels for this pixel
                    for (int c = 0; c < 3; c++)
                    {
                        // Direct pointer access for input pixel (HWC format)
                        byte pixelValue = imgPtrLocal[pixelIdx * 3 + c];
                        
                        // Calculate output position in CHW format: [channel][pixel]
                        float* outputPtr = tensorPtrLocal + c * imageSize + pixelIdx;
                        if (applyLowerHalfMask)
                        {
                            // Apply mask to raw pixel value first, then normalize
                            float maskedPixelValue = pixelValue * mask;
                            *outputPtr = maskedPixelValue * multipliersLocal[c] + offsetsLocal[c];
                        }
                        else
                        {
                            // No masking - direct normalization
                            *outputPtr = pixelValue * multipliersLocal[c] + offsetsLocal[c];
                        }
                    }
                });
            }
            
            return new DenseTensor<float>(tensorData, new[] { 1, 3, frame.height, frame.width });
        }
        
#endregion

#region Transformations

        /// <summary>
        /// Crop frame to specified rectangle
        /// </summary>
        /// <param name="frame">Frame to crop</param>
        /// <param name="cropRect">Rectangle to crop</param>
        /// <returns>Cropped frame</returns>
        public static unsafe Frame CropFrame(Frame frame, Rect cropRect)
        {
            if (frame.data.Length != frame.width * frame.height * 3)
                throw new ArgumentException($"Source data size {frame.data.Length} doesn't match expected size {frame.width * frame.height * 3} for RGB24 format");
            
            // Ensure crop bounds are within source image
            int x = Mathf.Max(0, (int)cropRect.x);
            int y = Mathf.Max(0, (int)cropRect.y);
            int cropWidth = Mathf.Min((int)cropRect.width, frame.width - x);
            int cropHeight = Mathf.Min((int)cropRect.height, frame.height - y);
            
            if (cropWidth <= 0 || cropHeight <= 0)
                throw new ArgumentException("Invalid crop rectangle");
            
            var croppedFrame = new Frame(new byte[cropWidth * cropHeight * 3], cropWidth, cropHeight);
            
            fixed (byte* srcPtr = frame.data)
            fixed (byte* dstPtr = croppedFrame.data)
            {
                // Capture pointers in local variables to avoid lambda closure issues
                byte* srcPtrLocal = srcPtr;
                byte* dstPtrLocal = dstPtr;
                
                System.Threading.Tasks.Parallel.For(0, cropHeight, row =>
                {
                    // Calculate source and destination row pointers
                    byte* srcRowPtr = srcPtrLocal + ((y + row) * frame.width + x) * 3; // RGB24: 3 bytes per pixel
                    byte* dstRowPtr = dstPtrLocal + row * cropWidth * 3;
                    
                    // Bulk copy entire row in one operation
                    int rowBytes = cropWidth * 3;
                    Buffer.MemoryCopy(srcRowPtr, dstRowPtr, rowBytes, rowBytes);
                });
            }
            
            return croppedFrame;
        }

        /// <summary>
        /// Resize frame to exact target dimensions (matching Python cv2.resize with LANCZOS4)
        /// </summary>
        /// <param name="frame">Frame to resize</param>
        /// <param name="targetWidth">Target width</param>
        /// <param name="targetHeight">Target height</param>
        /// <param name="samplingMode">Sampling mode</param>
        /// <returns>Resized frame</returns>
        public static unsafe Frame ResizeFrame(
            Frame frame, 
            int targetWidth, int targetHeight, 
            SamplingMode samplingMode = SamplingMode.Bilinear)
        {
            if (frame.data == null)
                throw new ArgumentNullException(nameof(frame.data));
                
            if (targetWidth <= 0 || targetHeight <= 0)
                throw new ArgumentException("Target dimensions must be positive");
                
            if (frame.width <= 0 || frame.height <= 0)
                throw new ArgumentException("Source dimensions must be positive");
                
            // If already the correct size, return copy
            if (frame.width == targetWidth && frame.height == targetHeight)
            {
                var copy = new byte[frame.data.Length];
                Array.Copy(frame.data, copy, frame.data.Length);
                return new Frame(copy, targetWidth, targetHeight);
            }
            
            // Create target byte array (RGB24 = 3 bytes per pixel)
            var targetFrame = new Frame(new byte[targetWidth * targetHeight * 3], targetWidth, targetHeight);
            
            // Pre-calculate scaling ratios for performance
            float xRatio = (float)frame.width / targetWidth;
            float yRatio = (float)frame.height / targetHeight;
            
            // OPTIMIZED: Maximum parallelism across all target pixels
            int totalPixels = targetWidth * targetHeight;
            
            if (samplingMode == SamplingMode.Point)
            {
                // FASTEST: Point/Nearest neighbor sampling - ~3-5x faster than bilinear
                System.Threading.Tasks.Parallel.For(0, totalPixels, pixelIndex =>
                {
                    // Calculate x, y coordinates from linear pixel index
                    int y = pixelIndex / targetWidth;
                    int x = pixelIndex % targetWidth;
                    
                    // Map target pixel to source coordinates and round to nearest
                    int srcX = Mathf.RoundToInt(x * xRatio);
                    int srcY = Mathf.RoundToInt(y * yRatio);
                    
                    // Clamp to source bounds
                    srcX = Mathf.Clamp(srcX, 0, frame.width - 1);
                    srcY = Mathf.Clamp(srcY, 0, frame.height - 1);
                    
                    // Calculate source and target indices
                    int sourceIdx = (srcY * frame.width + srcX) * 3;
                    int targetIdx = (y * targetWidth + x) * 3;
                    
                    // Direct 3-byte copy (RGB24)
                    targetFrame.data[targetIdx] = frame.data[sourceIdx];         // R
                    targetFrame.data[targetIdx + 1] = frame.data[sourceIdx + 1]; // G
                    targetFrame.data[targetIdx + 2] = frame.data[sourceIdx + 2]; // B
                });
            }
            else
            {
                // HIGH-QUALITY: Bilinear interpolation
                System.Threading.Tasks.Parallel.For(0, totalPixels, pixelIndex =>
                {
                    // Calculate x, y coordinates from linear pixel index
                    int y = pixelIndex / targetWidth;
                    int x = pixelIndex % targetWidth;
                    
                    // Map target pixel to source coordinates
                    float srcX = x * xRatio;
                    float srcY = y * yRatio;
                    
                    // Get integer and fractional parts for bilinear interpolation
                    int x1 = Mathf.FloorToInt(srcX);
                    int y1 = Mathf.FloorToInt(srcY);
                    int x2 = Mathf.Min(x1 + 1, frame.width - 1);
                    int y2 = Mathf.Min(y1 + 1, frame.height - 1);
                    
                    float fx = srcX - x1;
                    float fy = srcY - y1;
                    float invFx = 1.0f - fx;
                    float invFy = 1.0f - fy;
                    
                    // Calculate source pixel indices for bilinear interpolation
                    int c1Idx = (y1 * frame.width + x1) * 3; // Top-left
                    int c2Idx = (y1 * frame.width + x2) * 3; // Top-right
                    int c3Idx = (y2 * frame.width + x1) * 3; // Bottom-left
                    int c4Idx = (y2 * frame.width + x2) * 3; // Bottom-right
                    
                    // Pre-calculate bilinear interpolation weights
                    float w1 = invFx * invFy; // Top-left weight
                    float w2 = fx * invFy;    // Top-right weight
                    float w3 = invFx * fy;    // Bottom-left weight
                    float w4 = fx * fy;       // Bottom-right weight
                    
                    // Calculate target pixel index
                    int targetIdx = (y * targetWidth + x) * 3;
                    
                    // OPTIMIZED: Direct byte interpolation with unrolled RGB channels
                    // R channel
                    float r = frame.data[c1Idx] * w1 + frame.data[c2Idx] * w2 + frame.data[c3Idx] * w3 + frame.data[c4Idx] * w4;
                    targetFrame.data[targetIdx] = (byte)Mathf.Clamp(r, 0f, 255f);
                    
                    // G channel
                    float g = frame.data[c1Idx + 1] * w1 + frame.data[c2Idx + 1] * w2 + frame.data[c3Idx + 1] * w3 + frame.data[c4Idx + 1] * w4;
                    targetFrame.data[targetIdx + 1] = (byte)Mathf.Clamp(g, 0f, 255f);
                    
                    // B channel
                    float b = frame.data[c1Idx + 2] * w1 + frame.data[c2Idx + 2] * w2 + frame.data[c3Idx + 2] * w3 + frame.data[c4Idx + 2] * w4;
                    targetFrame.data[targetIdx + 2] = (byte)Mathf.Clamp(b, 0f, 255f);
                });
            }
            
            return targetFrame;
        }


        /// <summary>
        /// Affine transform frame to exact target dimensions (matching Python cv2.warpAffine)
        /// <param name="frame">Frame to transform</param>
        /// <param name="M">Transformation matrix</param>
        /// <param name="dstWidth">Target width</param>
        /// <param name="dstHeight">Target height</param>
        /// <returns>Transformed frame</returns>
        /// </summary>
        public static unsafe Frame AffineTransformFrame(
            Frame frame, float[,] M, int dstWidth, int dstHeight)
        {
            var result = new Frame(new byte[dstWidth * dstHeight * 3], dstWidth, dstHeight);

            int srcWidth = frame.width;
            int srcHeight = frame.height;

            // Invert the transformation matrix M to get the mapping from destination to source
            float[,] invM = MathUtils.InvertAffineTransform(M);
            
            // Pre-calculate matrix elements for performance (avoid repeated 2D array access)
            float m00 = invM[0, 0], m01 = invM[0, 1], m02 = invM[0, 2];
            float m10 = invM[1, 0], m11 = invM[1, 1], m12 = invM[1, 2];

            fixed (byte* resultPtr = result.data)
            {
                // Get source pointer using fixed for direct access
                fixed (byte* srcPtrFixed = frame.data)
                {
                    int totalPixels = dstWidth * dstHeight;
                    
                    // Capture pointers in local variables to avoid lambda closure issues
                    byte* srcPtrLocal = srcPtrFixed;
                    byte* resultPtrLocal = resultPtr;
                    
                    System.Threading.Tasks.Parallel.For(0, totalPixels, pixelIndex =>
                    {
                        // Calculate x, y coordinates from linear pixel index
                        int dstY = pixelIndex / dstWidth;
                        int dstX = pixelIndex % dstWidth;

                        // Apply inverse transformation matrix to find source coordinates
                        float srcX = m00 * dstX + m01 * dstY + m02;
                        float srcY = m10 * dstX + m11 * dstY + m12;

                        // Get integer and fractional parts for bilinear interpolation
                        int x0 = (int)srcX;
                        int y0 = (int)srcY;

                        float fx = srcX - x0;
                        float fy = srcY - y0;

                        // Default to black (borderValue=0.0 in OpenCV)
                        byte r = 0, g = 0, b = 0;

                        // Bounds check for bilinear interpolation
                        if (x0 >= 0 && (x0 + 1) < srcWidth && y0 >= 0 && (y0 + 1) < srcHeight)
                        {
                            byte* c00Ptr = srcPtrLocal + (y0 * srcWidth + x0) * 3;           // Top-left
                            byte* c10Ptr = srcPtrLocal + (y0 * srcWidth + x0 + 1) * 3;       // Top-right
                            byte* c01Ptr = srcPtrLocal + ((y0 + 1) * srcWidth + x0) * 3;     // Bottom-left
                            byte* c11Ptr = srcPtrLocal + ((y0 + 1) * srcWidth + x0 + 1) * 3; // Bottom-right

                            // Pre-calculate bilinear interpolation weights
                            float inv_fx = 1.0f - fx;
                            float inv_fy = 1.0f - fy;
                            float w00 = inv_fx * inv_fy; // Top-left weight
                            float w10 = fx * inv_fy;     // Top-right weight
                            float w01 = inv_fx * fy;     // Bottom-left weight
                            float w11 = fx * fy;         // Bottom-right weight
                            
                            float r_float = w00 * c00Ptr[0] + w10 * c10Ptr[0] + w01 * c01Ptr[0] + w11 * c11Ptr[0];
                            float g_float = w00 * c00Ptr[1] + w10 * c10Ptr[1] + w01 * c01Ptr[1] + w11 * c11Ptr[1];
                            float b_float = w00 * c00Ptr[2] + w10 * c10Ptr[2] + w01 * c01Ptr[2] + w11 * c11Ptr[2];

                            r = (byte)(r_float < 0f ? 0 : r_float > 255f ? 255 : r_float);
                            g = (byte)(g_float < 0f ? 0 : g_float > 255f ? 255 : g_float);
                            b = (byte)(b_float < 0f ? 0 : b_float > 255f ? 255 : b_float);
                        }

                        byte* resultPixelPtr = resultPtrLocal + pixelIndex * 3;
                        resultPixelPtr[0] = r; // R
                        resultPixelPtr[1] = g; // G
                        resultPixelPtr[2] = b; // B
                    });
                }
            }

            return result;
        }
        
        
#endregion

#region Image Processing

        /// <summary>
        /// Unsafe optimized morphological operation using direct byte pointer access
        /// OPTIMIZED: Uses parallelization and direct memory operations for maximum performance
        /// Consolidates dilation and erosion into a single optimized function
        /// </summary>
        /// <param name="inputPtr">Input image pointer</param>
        /// <param name="outputPtr">Output image pointer</param>
        /// <param name="width">Image width</param>
        /// <param name="height">Image height</param>
        /// <param name="kernelSize">Kernel size (odd number)</param>
        /// <param name="operation">Morphological operation type</param>
        private static unsafe void ApplyMorphologyUnsafe(
            byte* inputPtr, byte* outputPtr, int width, int height, int kernelSize, MorphologyOperation operation)
        {
            int radius = kernelSize / 2;
            
            // Pre-calculate operation-specific values for performance
            byte initialValue = operation == MorphologyOperation.Dilation ? (byte)0 : (byte)255;
            bool isDilation = operation == MorphologyOperation.Dilation;
            
            System.Threading.Tasks.Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    byte resultValue = initialValue;
                    
                    // Check kernel neighborhood
                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            int ny = Mathf.Clamp(y + ky, 0, height - 1);
                            int nx = Mathf.Clamp(x + kx, 0, width - 1);
                            
                            // Get pixel pointer and use red channel as grayscale value
                            byte* samplePixel = inputPtr + (ny * width + nx) * 3;
                            byte sampleValue = samplePixel[0]; // Use red channel
                            
                            // Apply operation-specific logic
                            if (isDilation)
                            {
                                resultValue = (byte)Mathf.Max(resultValue, sampleValue);
                            }
                            else
                            {
                                resultValue = (byte)Mathf.Min(resultValue, sampleValue);
                            }
                        }
                    }
                    
                    // Set output pixel (RGB24: all channels same for grayscale)
                    byte* outputPixel = outputPtr + (y * width + x) * 3;
                    outputPixel[0] = resultValue; // R
                    outputPixel[1] = resultValue; // G
                    outputPixel[2] = resultValue; // B
                }
            });
        }
        
        /// <summary>
        /// Unsafe optimized dilation operation using direct byte pointer access
        /// OPTIMIZED: Wrapper around consolidated ApplyMorphologyUnsafe for backward compatibility
        /// </summary>
        public static unsafe Frame ApplyDilation(
            Frame frame, int kernelSize)
        {
            var dilatedFrame = new Frame(new byte[frame.data.Length], frame.width, frame.height);
            fixed (byte* inputPtr = frame.data)
            fixed (byte* outputPtr = dilatedFrame.data)
            {
                ApplyMorphologyUnsafe(inputPtr, outputPtr, frame.width, frame.height, kernelSize, MorphologyOperation.Dilation);
            }
            return dilatedFrame;
        }
        
        /// <summary>
        /// Unsafe optimized erosion operation using direct byte pointer access
        /// OPTIMIZED: Wrapper around consolidated ApplyMorphologyUnsafe for backward compatibility
        /// </summary>
        public static unsafe Frame ApplyErosion(
            Frame frame, int kernelSize)
        {
            var erodedFrame = new Frame(new byte[frame.data.Length], frame.width, frame.height);
            fixed (byte* inputPtr = frame.data)
            fixed (byte* outputPtr = erodedFrame.data)
            {
                ApplyMorphologyUnsafe(inputPtr, outputPtr, frame.width, frame.height, kernelSize, MorphologyOperation.Erosion);
            }
            return erodedFrame;
        }

        /// <summary>
        /// Consolidated blur pass function to eliminate code duplication
        /// OPTIMIZED: Single unified implementation for both horizontal and vertical passes
        /// REFACTORED: Eliminates if/else duplication by abstracting coordinate calculations
        /// </summary>
        /// <param name="inputPtr">Input image pointer</param>
        /// <param name="outputPtr">Output image pointer</param>
        /// <param name="width">Image width</param>
        /// <param name="height">Image height</param>
        /// <param name="kernel">Normalized Gaussian kernel weights</param>
        /// <param name="direction">Blur direction (horizontal or vertical)</param>
        private static unsafe void ApplyBlurPass(
            byte* inputPtr, byte* outputPtr, int width, int height, float[] kernel, BlurDirection direction)
        {
            int kernelSize = kernel.Length;
            int halfKernel = kernelSize / 2;
            bool isHorizontal = direction == BlurDirection.Horizontal;
            
            // Direction-specific parameters
            int parallelCount = isHorizontal ? height : width;  // What to parallelize over
            int innerCount = isHorizontal ? width : height;     // Inner loop count
            int maxCoord = isHorizontal ? width - 1 : height - 1; // Max coordinate for clamping
            
            // UNIFIED: Single parallel loop that works for both directions
            System.Threading.Tasks.Parallel.For(0, parallelCount, outerIndex =>
            {
                for (int innerIndex = 0; innerIndex < innerCount; innerIndex++)
                {
                    float r = 0, g = 0, b = 0;
                    
                    // Apply kernel along the blur axis
                    for (int k = 0; k < kernelSize; k++)
                    {
                        // Calculate sample coordinate with clamping
                        int sampleCoord = Mathf.Clamp(innerIndex + k - halfKernel, 0, maxCoord);
                        
                        // Calculate pixel pointer based on direction
                        RGB24* samplePixel;
                        if (isHorizontal)
                        {
                            // Horizontal: outerIndex=y, innerIndex=x, sampleCoord=sampleX
                            samplePixel = (RGB24*)(inputPtr + outerIndex * width * 3) + sampleCoord;
                        }
                        else
                        {
                            // Vertical: outerIndex=x, innerIndex=y, sampleCoord=sampleY
                            samplePixel = (RGB24*)(inputPtr + sampleCoord * width * 3) + outerIndex;
                        }
                        
                        float weight = kernel[k];
                        
                        // Accumulate weighted RGB values
                        r += samplePixel->r * weight;
                        g += samplePixel->g * weight;
                        b += samplePixel->b * weight;
                    }
                    
                    // Calculate output pixel pointer based on direction
                    RGB24* targetPixel;
                    if (isHorizontal)
                    {
                        // Horizontal: outerIndex=y, innerIndex=x
                        targetPixel = (RGB24*)(outputPtr + outerIndex * width * 3) + innerIndex;
                    }
                    else
                    {
                        // Vertical: outerIndex=x, innerIndex=y
                        targetPixel = (RGB24*)(outputPtr + innerIndex * width * 3) + outerIndex;
                    }
                    
                    // Write result to output
                    targetPixel->r = (byte)Mathf.Clamp(r, 0, 255);
                    targetPixel->g = (byte)Mathf.Clamp(g, 0, 255);
                    targetPixel->b = (byte)Mathf.Clamp(b, 0, 255);
                }
            });
        }

        /// <summary>
        /// Unsafe optimized Gaussian blur implementation using raw pointers
        /// UNSAFE OPTIMIZED: Direct pointer-to-pointer processing for maximum performance
        /// REFACTORED: Uses consolidated ApplyBlurPass to eliminate code duplication
        /// </summary>
        public static unsafe Frame ApplySimpleGaussianBlur(Frame frame, int kernelSize)
        {
            if (frame.data == null)
            {
                Debug.LogError("[FrameUtils] Null frame passed to Gaussian blur");
                return frame;
            }
            
            if (frame.width <= 0 || frame.height <= 0)
            {
                Debug.LogError($"[FrameUtils] Invalid dimensions: {frame.width}x{frame.height}");
                return frame;
            }
            
            float sigma = kernelSize / 3.0f;
            int halfKernel = kernelSize / 2;
            
            // Generate 1D Gaussian kernel (same for both directions)
            var kernel = new float[kernelSize];
            float sum = 0;
            
            for (int i = 0; i < kernelSize; i++)
            {
                int offset = i - halfKernel;
                kernel[i] = Mathf.Exp(-(offset * offset) / (2 * sigma * sigma));
                sum += kernel[i];
            }
            
            // Normalize kernel
            for (int i = 0; i < kernelSize; i++)
            {
                kernel[i] /= sum;
            }
            
            // Allocate temporary buffer for horizontal pass
            int totalPixels = frame.width * frame.height;
            byte* tempPtr = (byte*)UnsafeUtility.Malloc(totalPixels * 3, 4, Unity.Collections.Allocator.Temp);
            var outputFrame = new Frame(new byte[totalPixels * 3], frame.width, frame.height);
            try
            {
                fixed (byte* inputPtr = frame.data)
                fixed (byte* outputPtr = outputFrame.data)
                {
                    // Step 1: Horizontal blur pass (input -> temp)
                    ApplyBlurPass(inputPtr, tempPtr, frame.width, frame.height, kernel, BlurDirection.Horizontal);
                
                    // Step 2: Vertical blur pass (temp -> output)
                    ApplyBlurPass(tempPtr, outputPtr, frame.width, frame.height, kernel, BlurDirection.Vertical);
                }
            }   
            finally
            {
                // Clean up temporary buffer
                UnsafeUtility.Free(tempPtr, Unity.Collections.Allocator.Temp);
            }
            return outputFrame;
        }

#endregion

    }
}
