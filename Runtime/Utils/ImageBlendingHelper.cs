using UnityEngine;
using Unity.Collections.LowLevel.Unsafe;

namespace MuseTalk.Utils
{
    /// <summary>
    /// Image blending helper that implements Python's get_image() functionality
    /// Provides seamless face composition with mask-based blending
    /// </summary>
    public static class ImageBlendingHelper
    {
        /// <summary>
        /// Blend face with original image using cached segmentation mask
        /// REFACTORED: Now works with byte arrays for better memory efficiency
        /// OPTIMIZED: Uses pre-computed segmentation data to avoid regenerating mask for every frame
        /// </summary>
        public static byte[] BlendFaceWithOriginal(
            byte[] originalImage, int originalWidth, int originalHeight, 
            byte[] faceTexture, int faceWidth, int faceHeight,  
            Vector4 faceBbox, Vector4 cropBox,
            byte[] precomputedBlurredMask, int precomputedBlurredMaskWidth, int precomputedBlurredMaskHeight,
            byte[] precomputedFaceLarge, int precomputedFaceLargeWidth, int precomputedFaceLargeHeight, 
            string mode = "raw")
        {
            if (precomputedBlurredMask != null)
            {
                // OPTIMAL PATH: Use precomputed blurred mask for maximum performance
                Vector4 adjustedFaceBbox = faceBbox;
                if (mode == "jaw") // v15 mode
                {
                    adjustedFaceBbox.w = Mathf.Min(adjustedFaceBbox.w + 10f, originalHeight);
                }
                
                // Apply the precomputed blurred mask to blend the face
                var result = ApplySegmentationMaskWithPrecomputedMasks(
                    originalImage, originalWidth, originalHeight, 
                    faceTexture, faceWidth, faceHeight, 
                    adjustedFaceBbox, cropBox,
                    precomputedBlurredMask, precomputedBlurredMaskWidth, precomputedBlurredMaskHeight,
                    precomputedFaceLarge, precomputedFaceLargeWidth, precomputedFaceLargeHeight, 
                    mode);
                return result;
            }
            else
            {
                throw new System.Exception("Precomputed blurred mask is null");
            }
        }

        /// <summary>
        /// Crop image to specified rectangle
        /// REFACTORED: Now works with byte arrays for better memory efficiency
        /// </summary>
        private static (byte[], int, int) CropImage(byte[] sourceData, int sourceWidth, int sourceHeight, Rect cropRect)
        {
            // Ensure crop bounds are within image
            cropRect.x = Mathf.Max(0, cropRect.x);
            cropRect.y = Mathf.Max(0, cropRect.y);
            cropRect.width = Mathf.Min(cropRect.width, sourceWidth - cropRect.x);
            cropRect.height = Mathf.Min(cropRect.height, sourceHeight - cropRect.y);
            
            return (TextureUtils.CropTexture(sourceData, sourceWidth, sourceHeight, cropRect), 
                    (int)cropRect.width, (int)cropRect.height);
        }

        /// <summary>
        /// Create mask_small from BiSeNet result by cropping to face region
        /// REFACTORED: Now works with byte arrays for better memory efficiency
        /// Matches Python: mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
        /// </summary>
        public static (byte[], int, int) CreateSmallMask(
            byte[] maskData, int maskWidth, int maskHeight,
            Vector4 faceBbox, Rect cropBox)
        {
            float x = faceBbox.x;
            float y = faceBbox.y;
            float x1 = faceBbox.z;
            float y1 = faceBbox.w;
            float x_s = cropBox.x;
            float y_s = cropBox.y;
            
            // Calculate crop region in BiSeNet mask space
            Rect cropRect = new(
                x - x_s,
                y - y_s,
                x1 - x,  // width = x1 - x
                y1 - y   // height = y1 - y
            );
            
            return CropImage(maskData, maskWidth, maskHeight, cropRect);
        }

        /// <summary>
        /// Create full mask by pasting small mask into blank canvas (matching Python mask_full)
        /// REFACTORED: Now works with byte arrays for better memory efficiency
        /// OPTIMIZED: Uses unsafe pointers, parallelization, and memcpy for maximum performance
        /// </summary>
        public static unsafe (byte[], int, int) CreateFullMask(
            byte[] originalMaskData, int originalWidth, int originalHeight, 
            byte[] smallMaskData, int smallWidth, int smallHeight, 
            Vector4 faceBbox, Rect cropBox)
        {
            // Python: mask_image = Image.new('L', ori_shape, 0)
            // Python: mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))
            
            int width = originalWidth;
            int height = originalHeight;
            int rowSizeBytes = width * 3; // RGB24: 3 bytes per pixel
            
            // Create blank mask data with RGB24 format for efficiency
            var fullMaskData = new byte[width * height * 3];
            
            // Calculate paste position (matching Python coordinates)
            float x = faceBbox.x;
            float y = faceBbox.y;
            float x_s = cropBox.x;
            float y_s = cropBox.y;
            
            int pasteX = Mathf.RoundToInt(x - x_s);
            int pasteY = Mathf.RoundToInt(y - y_s);
            
            // Calculate valid paste region to avoid bounds checking in inner loop
            int startX = Mathf.Max(0, pasteX);
            int endX = Mathf.Min(width, pasteX + smallWidth);
            int startY = Mathf.Max(0, pasteY);
            int endY = Mathf.Min(height, pasteY + smallHeight);
            
            // Use unsafe pointers for maximum performance with memcpy operations
            fixed (byte* fullMaskPtr = fullMaskData)
            fixed (byte* smallMaskPtr = smallMaskData)
            {
                byte* fullMaskPtrLocal = fullMaskPtr;
                byte* smallMaskPtrLocal = smallMaskPtr;
                
                // Initialize to black using parallel memclear (much faster than loops)
                System.Threading.Tasks.Parallel.For(0, height, y =>
                {
                    byte* targetRowPtr = fullMaskPtrLocal + y * rowSizeBytes;
                    UnsafeUtility.MemClear(targetRowPtr, rowSizeBytes);
                });
                
                // Parallel paste operation - process rows concurrently
                System.Threading.Tasks.Parallel.For(startY, endY, targetY =>
                {
                    int sourceY = targetY - pasteY;
                    if (sourceY >= 0 && sourceY < smallHeight)
                    {
                        byte* targetRowPtr = fullMaskPtrLocal + targetY * rowSizeBytes;
                        byte* sourceRowPtr = smallMaskPtrLocal + sourceY * smallWidth * 3;
                        
                        // Check if we can do a full row copy (when small mask width matches and aligns perfectly)
                        if (pasteX == 0 && startX == 0 && endX == width && smallWidth == width)
                        {
                            // Bulk copy entire row using native memcpy (fastest path)
                            UnsafeUtility.MemCpy(targetRowPtr, sourceRowPtr, smallWidth * 3);
                        }
                        else
                        {
                            // Partial row copy - process pixels in chunks when possible
                            for (int targetX = startX; targetX < endX; targetX++)
                            {
                                int sourceX = targetX - pasteX;
                                if (sourceX >= 0 && sourceX < smallWidth)
                                {
                                    byte* targetPixelPtr = targetRowPtr + targetX * 3;
                                    byte* sourcePixelPtr = sourceRowPtr + sourceX * 3;
                                    
                                    // Use red channel as grayscale value for all RGB components
                                    byte grayValue = sourcePixelPtr[0]; // Red channel
                                    targetPixelPtr[0] = grayValue; // R
                                    targetPixelPtr[1] = grayValue; // G
                                    targetPixelPtr[2] = grayValue; // B
                                }
                            }
                        }
                    }
                });
            }
            
            return (fullMaskData, width, height);
        }

        /// <summary>
        /// Apply upper boundary ratio to preserve upper face area (matching Python)
        /// REFACTORED: Now works with byte arrays for better memory efficiency
        /// OPTIMIZED: Uses unsafe pointers, parallelization, and memcpy for maximum performance
        /// </summary>
        public static unsafe (byte[], int, int) ApplyUpperBoundaryRatio(byte[] maskData, int width, int height, float upperBoundaryRatio)
        {
            // Create result data with RGB24 format for efficiency
            var resultData = new byte[width * height * 3];
            
            // Calculate boundary line (matching Python upper_boundary_ratio logic)
            // In standard image coordinates: Y=0 is top, Y=height-1 is bottom
            // upperBoundaryRatio=0.5 means preserve top 50% of face (upper half)
            // Python: top_boundary = int(height * upper_boundary_ratio)
            int boundaryY = Mathf.RoundToInt(height * upperBoundaryRatio);
            int rowSizeBytes = width * 3; // RGB24: 3 bytes per pixel
            
            // Use unsafe pointers for maximum performance with memcpy operations
            fixed (byte* maskPtr = maskData)
            fixed (byte* resultPtr = resultData)
            {
                byte* maskPtrLocal = maskPtr;
                byte* resultPtrLocal = resultPtr;
                
                // Parallel processing of rows for maximum performance
                System.Threading.Tasks.Parallel.For(0, height, y =>
                {
                    byte* sourceRowPtr = maskPtrLocal + y * rowSizeBytes;
                    byte* targetRowPtr = resultPtrLocal + y * rowSizeBytes;
                    
                    // In standard image coordinates: lower Y = upper part of image
                    if (y < boundaryY) // Upper part of face (preserve original - eyes, nose, forehead)
                    {
                        // Zero out entire row using fast memset (preserve original face - eyes, nose, forehead)
                        UnsafeUtility.MemClear(targetRowPtr, rowSizeBytes);
                    }
                    else // Lower part of face (talking area - mouth, chin)
                    {
                        // Copy entire row from mask to result using fast memcpy (allow blending)
                        UnsafeUtility.MemCpy(targetRowPtr, sourceRowPtr, rowSizeBytes);
                    }
                });
            }
            
            return (resultData, width, height);
        }

        /// <summary>
        /// Paste generated face into the large face region
        /// OPTIMIZED: Uses unsafe pointers, parallelization, and memcpy for maximum performance
        /// </summary>
        private static unsafe byte[] PasteFaceIntoLarge(
            byte[] faceLarge, int faceLargeWidth, int faceLargeHeight,
            byte[] generatedFace, int generatedFaceWidth, int generatedFaceHeight,
            Vector4 faceBbox,
            Rect cropBox)
        {
            // Calculate relative position within face_large (image coordinates) - use exact integer calculation like Python
            int relativeX = (int)(faceBbox.x - cropBox.x);
            int relativeY = (int)(faceBbox.y - cropBox.y);
            
            // Create result texture with RGB24 format for efficiency
            var result = new byte[faceLargeWidth * faceLargeHeight * 3];
            var faceLargePixelData = faceLarge;
            var generatedFacePixelData = generatedFace;
            var resultPixelData = result;
            
            // Get unsafe pointers for direct memory operations
            fixed (byte* faceLargePtr = faceLargePixelData)
            fixed (byte* generatedFacePtr = generatedFacePixelData)
            fixed (byte* resultPtr = resultPixelData)
            {
                int resultWidth = faceLargeWidth;
                int resultHeight = faceLargeHeight;
                int faceWidth = generatedFaceWidth;
                int faceHeight = generatedFaceHeight;

                byte* faceLargePtrLocal = faceLargePtr;
                byte* generatedFacePtrLocal = generatedFacePtr;
                byte* resultPtrLocal = resultPtr;

                // First, copy the entire faceLarge to result using parallel memcpy
                System.Threading.Tasks.Parallel.For(0, resultHeight, y =>
                {
                    byte* sourceRowPtr = faceLargePtrLocal + y * resultWidth * 3;
                    byte* targetRowPtr = resultPtrLocal + y * resultWidth * 3;
                    UnsafeUtility.MemCpy(targetRowPtr, sourceRowPtr, resultWidth * 3);
                });

                // Calculate valid paste region to avoid bounds checking in inner loop
                int startX = Mathf.Max(0, relativeX);
                int endX = Mathf.Min(resultWidth, relativeX + faceWidth);
                int startY = Mathf.Max(0, relativeY);
                int endY = Mathf.Min(resultHeight, relativeY + faceHeight);
                
                // Parallel paste operation - process rows of the generated face
                System.Threading.Tasks.Parallel.For(startY, endY, targetY =>
                {
                    int sourceY = targetY - relativeY;
                    if (sourceY >= 0 && sourceY < faceHeight)
                    {
                        byte* targetRowPtr = resultPtrLocal + targetY * resultWidth * 3;
                        byte* sourceRowPtr = generatedFacePtrLocal + sourceY * faceWidth * 3;
                        
                        // Check if we can do a full row copy (when face width matches and aligns perfectly)
                        if (relativeX == 0 && startX == 0 && endX == resultWidth && faceWidth == resultWidth)
                        {
                            // Bulk copy entire row using native memcpy (fastest path)
                            UnsafeUtility.MemCpy(targetRowPtr, sourceRowPtr, faceWidth * 3);
                        }
                        else
                        {
                            // Partial row copy - use optimized chunk copying when possible
                            int copyStartX = startX;
                            int copyEndX = endX;
                            int sourceStartX = copyStartX - relativeX;
                            int copyWidth = copyEndX - copyStartX;
                            
                            if (copyWidth > 0 && sourceStartX >= 0)
                            {
                                byte* targetPtr = targetRowPtr + copyStartX * 3;
                                byte* sourcePtr = sourceRowPtr + sourceStartX * 3;
                                
                                // Use memcpy for contiguous pixel chunks (much faster than pixel-by-pixel)
                                UnsafeUtility.MemCpy(targetPtr, sourcePtr, copyWidth * 3);
                            }
                        }
                    }
                });
            }
            return result;
        }
        
        /// <summary>
        /// Apply segmentation mask to blend face with original image using precomputed masks
        /// OPTIMIZED: Uses precomputed masks for maximum performance
        /// </summary>
        private static byte[] ApplySegmentationMaskWithPrecomputedMasks(
            byte[] originalImage, int originalWidth, int originalHeight, 
            byte[] faceTexture, int faceTextureWidth, int faceTextureHeight,  
            Vector4 faceBbox, Vector4 cropBox,
            byte[] precomputedBlurredMask, int precomputedBlurredMaskWidth, int precomputedBlurredMaskHeight,
            byte[] precomputedFaceLarge, int precomputedFaceLargeWidth, int precomputedFaceLargeHeight, 
            string mode = "raw")
        {
            // Step 1: Use precomputed face_large and paste the resized face into it (matching Python exactly)
            // Python: face_large = body.crop(crop_box); face_large.paste(face, (x-x_s, y-y_s))
            // Note: faceLarge is already precomputed, no need to crop again
            
            // Resize face texture to match face bbox dimensions (use exact integer calculation like Python)
            int faceWidth = (int)(faceBbox.z - faceBbox.x);
            int faceHeight = (int)(faceBbox.w - faceBbox.y);
            var resizedFace = TextureUtils.ResizeTextureToExactSize(faceTexture, faceTextureWidth, faceTextureHeight, faceWidth, faceHeight);

            // Paste the resized face into precomputed face_large at relative position (matching Python)
            // Python: face_large.paste(face, (x-x_s, y-y_s))
            var cropRect = new Rect(cropBox.x, cropBox.y, cropBox.z - cropBox.x, cropBox.w - cropBox.y);
            var faceLargeWithFace = PasteFaceIntoLarge(
                precomputedFaceLarge, precomputedFaceLargeWidth, precomputedFaceLargeHeight, 
                resizedFace, faceWidth, faceHeight, 
                faceBbox, cropRect);
            // Step 2: Composite images using the precomputed blurred mask (matching Python alpha blending)
            // Python: body.paste(face_large, crop_box[:2], mask_image)
            var result = CompositeWithMask(
                originalImage, originalWidth, originalHeight,
                faceLargeWithFace, precomputedFaceLargeWidth, precomputedFaceLargeHeight,
                faceBbox, 
                precomputedBlurredMask, precomputedBlurredMaskWidth, precomputedBlurredMaskHeight,
                cropBox);

            return result;
        }
        
        /// <summary>
        /// Apply Gaussian blur to mask for smooth blending (matching Python)
        /// REFACTORED: Now works with byte arrays for better memory efficiency
        /// </summary>
        public static unsafe (byte[], int, int) ApplyGaussianBlurToMask(byte[] maskData, int width, int height)
        {
            // Calculate blur kernel size based on mask dimensions (matching Python)
            float blurFactor = 0.08f; // jaw mode blur factor from Python
            int kernelSize = Mathf.RoundToInt(blurFactor * width / 2) * 2 + 1;
            kernelSize = Mathf.Max(kernelSize, 15); // Minimum kernel size

            byte[] blurredMaskData = new byte[maskData.Length];
            fixed (byte* maskPtr = maskData)
            fixed (byte* blurredMaskPtr = blurredMaskData)
            {
                TextureUtils.ApplySimpleGaussianBlur(maskPtr, blurredMaskPtr, width, height, kernelSize);
            }
            
            return (blurredMaskData, width, height);
        }
        
        /// <summary>
        /// Composite images using blurred mask (matching Python exactly)
        /// OPTIMIZED: Uses unsafe pointers, parallelization, and optimized alpha blending
        /// Python: body.paste(face_large, crop_box[:2], mask_image)
        /// </summary>
        private static unsafe byte[] CompositeWithMask(
            byte[] originalImage, int originalWidth, int originalHeight,
            byte[] faceLarge, int faceLargeWidth, int faceLargeHeight,
            Vector4 faceBbox,
            byte[] blurredMask, int blurredMaskWidth, int blurredMaskHeight,
            Vector4 cropBox)
        {
            int resultWidth = originalWidth;
            int resultHeight = originalHeight;
            int blendWidth = faceLargeWidth;
            int blendHeight = faceLargeHeight;
            
            // Create result texture with RGB24 format for efficiency
            var result = new byte[resultWidth * resultHeight * 3];
            var originalPixelData = originalImage;
            var faceLargePixelData = faceLarge;
            var maskPixelData = blurredMask;
            var resultPixelData = result;
            
            // Get unsafe pointers for direct memory operations
            fixed (byte* originalPtr = originalPixelData)
            fixed (byte* faceLargePtr = faceLargePixelData)
            fixed (byte* maskPtr = maskPixelData)
            fixed (byte* resultPtr = resultPixelData)
            {
                byte* originalPtrLocal = originalPtr;
                byte* faceLargePtrLocal = faceLargePtr;
                byte* maskPtrLocal = maskPtr;
                byte* resultPtrLocal = resultPtr;

                // Calculate paste position (matching Python: crop_box[:2])
                int pasteX = (int)cropBox.x;
                int pasteY = (int)cropBox.y;
                
                // First, copy the entire original image to result using parallel memcpy
                System.Threading.Tasks.Parallel.For(0, resultHeight, y =>
                {
                    byte* sourceRowPtr = originalPtrLocal + y * resultWidth * 3;
                    byte* targetRowPtr = resultPtrLocal + y * resultWidth * 3;
                    UnsafeUtility.MemCpy(targetRowPtr, sourceRowPtr, resultWidth * 3);
                });
                
                // Calculate valid blend region to avoid bounds checking in inner loop
                int startX = Mathf.Max(0, pasteX);
                int endX = Mathf.Min(resultWidth, pasteX + blendWidth);
                int startY = Mathf.Max(0, pasteY);
                int endY = Mathf.Min(resultHeight, pasteY + blendHeight);
                
                // Parallel alpha blending - process rows concurrently
                System.Threading.Tasks.Parallel.For(startY, endY, targetY =>
                {
                    int sourceY = targetY - pasteY;
                    if (sourceY >= 0 && sourceY < blendHeight)
                    {
                        byte* resultRowPtr = resultPtrLocal + targetY * resultWidth * 3;
                        byte* faceLargeRowPtr = faceLargePtrLocal + sourceY * blendWidth * 3;
                        byte* maskRowPtr = maskPtrLocal + sourceY * blendWidth * 3;
                        
                        // Process pixels in this row
                        for (int targetX = startX; targetX < endX; targetX++)
                        {
                            int sourceX = targetX - pasteX;
                            if (sourceX >= 0 && sourceX < blendWidth)
                            {
                                byte* targetPixel = resultRowPtr + targetX * 3;
                                byte* sourcePixel = faceLargeRowPtr + sourceX * 3;
                                byte* maskPixel = maskRowPtr + sourceX * 3;
                                
                                // Get mask alpha (use red channel as alpha, convert to 0-1 range)
                                float alpha = maskPixel[0] / 255.0f;
                                
                                if (alpha > 0.001f) // Small threshold to avoid unnecessary blending
                                {
                                    // Optimized alpha blend: result = foreground * alpha + background * (1 - alpha)
                                    float invAlpha = 1.0f - alpha;
                                    
                                    // Blend RGB channels directly with byte arithmetic
                                    targetPixel[0] = (byte)(sourcePixel[0] * alpha + targetPixel[0] * invAlpha); // R
                                    targetPixel[1] = (byte)(sourcePixel[1] * alpha + targetPixel[1] * invAlpha); // G
                                    targetPixel[2] = (byte)(sourcePixel[2] * alpha + targetPixel[2] * invAlpha); // B
                                }
                            }
                        }
                    }
                });
            }
            
            return result;
        }
    }
} 