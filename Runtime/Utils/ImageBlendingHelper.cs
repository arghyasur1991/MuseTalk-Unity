using System;
using UnityEngine;
using Unity.Collections.LowLevel.Unsafe;
using System.Runtime.InteropServices;

namespace MuseTalk.Utils
{
    using Models;
    /// <summary>
    /// RGB24 pixel struct for efficient 3-byte operations
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct RGB24
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
    /// Image blending helper that implements Python's get_image() functionality
    /// Provides seamless face composition with mask-based blending
    /// </summary>
    public static class ImageBlendingHelper
    {
        private static readonly DebugLogger Logger = new();
        private static FaceParsingHelper _FaceParsingHelper;
        
        /// <summary>
        /// Get or create the ONNX face parsing helper instance (singleton pattern)
        /// </summary>
        public static FaceParsingHelper GetOrCreateFaceParsingHelper(MuseTalkConfig config)
        {
            if (_FaceParsingHelper == null)
            {
                try
                {
                    _FaceParsingHelper = new FaceParsingHelper(config);
                }
                catch (Exception e)
                {
                    Logger.LogError($"[ImageBlendingHelper] Failed to initialize ONNX Face Parsing: {e.Message}");
                    _FaceParsingHelper = null;
                }
            }
            return _FaceParsingHelper;
        }
        
        /// <summary>
        /// Extract face region from original image using bounding box
        /// </summary>
        private static Texture2D ExtractFaceRegion(Texture2D originalImage, Vector4 faceBbox)
        {
            int x1 = Mathf.Max(0, (int)faceBbox.x);
            int y1 = Mathf.Max(0, (int)faceBbox.y);
            int x2 = Mathf.Min(originalImage.width, (int)faceBbox.z);
            int y2 = Mathf.Min(originalImage.height, (int)faceBbox.w);
            
            int faceWidth = x2 - x1;
            int faceHeight = y2 - y1;
            
            if (faceWidth <= 0 || faceHeight <= 0)
            {
                Logger.LogError("[ImageBlendingHelper] Invalid face bbox for extraction");
                return null;
            }
            
            var faceRegion = new Texture2D(faceWidth, faceHeight, TextureFormat.RGBA32, false);
            var pixels = originalImage.GetPixels(x1, y1, faceWidth, faceHeight);
            faceRegion.SetPixels(pixels);
            faceRegion.Apply();
            
            return faceRegion;
        }

        /// <summary>
        /// Blend face with original image using cached segmentation mask
        /// OPTIMIZED: Uses pre-computed segmentation data to avoid regenerating mask for every frame
        /// </summary>
        public static Texture2D BlendFaceWithOriginal(Texture2D originalImage, Texture2D faceTexture,
            Vector4 faceBbox, string mode = "raw", Vector4 cachedCropBox = default,
            Texture2D precomputedBlurredMask = null, Texture2D precomputedFaceLarge = null)
        {
            try
            {
                // Use precomputed blurred mask if available (optimal path)
                if (precomputedBlurredMask != null && cachedCropBox != default)
                {
                    // OPTIMAL PATH: Use precomputed blurred mask for maximum performance
                    Vector4 adjustedFaceBbox = faceBbox;
                    if (mode == "jaw") // v15 mode
                    {
                        adjustedFaceBbox.w = Mathf.Min(adjustedFaceBbox.w + 10f, originalImage.height);
                    }
                    
                    // Apply the precomputed blurred mask to blend the face
                    var result = ApplySegmentationMaskWithPrecomputedMasks(originalImage, faceTexture, adjustedFaceBbox, cachedCropBox, precomputedBlurredMask, precomputedFaceLarge);
                    return result;
                }
                else
                {                    
                    // No fallback - throw exception if face segmentation fails
                    throw new InvalidOperationException("Face segmentation failed and no fallback is available");
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"[ImageBlendingHelper] Blending failed: {e.Message}");
                throw;
            }
        }

        /// <summary>
        /// Calculate crop box with expansion factor (matching Python get_crop_box)
        /// </summary>
        private static Rect GetCropBox(Vector4 faceBbox, float expandFactor)
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
        private static Texture2D CropImage(Texture2D source, Rect cropRect)
        {
            // Ensure crop bounds are within image
            cropRect.x = Mathf.Max(0, cropRect.x);
            cropRect.y = Mathf.Max(0, cropRect.y);
            cropRect.width = Mathf.Min(cropRect.width, source.width - cropRect.x);
            cropRect.height = Mathf.Min(cropRect.height, source.height - cropRect.y);
            
            return TextureUtils.CropTexture(source, cropRect);
        }

        /// <summary>
        /// Create mask_small from BiSeNet result by cropping to face region
        /// Matches Python: mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
        /// </summary>
        public static Texture2D CreateSmallMaskFromBiSeNet(Texture2D biSeNetMask, Vector4 faceBbox, Rect cropBox)
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
            
            return CropImage(biSeNetMask, cropRect);
        }
        
        /// <summary>
        /// Paste mask_small into blank canvas at correct position
        /// Matches Python: mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))
        /// </summary>
        private static Texture2D PasteMaskSmallIntoBlank(Texture2D blankMask, Texture2D maskSmall, Vector4 faceBbox, Rect cropBox)
        {
            float x = faceBbox.x;
            float y = faceBbox.y;
            float x_s = cropBox.x;
            float y_s = cropBox.y;
            
            int pasteX = Mathf.RoundToInt(x - x_s);
            int pasteY = Mathf.RoundToInt(y - y_s);
            
            // Create result texture
            var result = new Texture2D(blankMask.width, blankMask.height, TextureFormat.R8, false);
            var resultPixels = blankMask.GetRawTextureData<byte>().ToArray();
            var smallPixels = maskSmall.GetRawTextureData<byte>();
            
            // Paste mask_small into result at calculated position
            for (int sy = 0; sy < maskSmall.height; sy++)
            {
                for (int sx = 0; sx < maskSmall.width; sx++)
                {
                    int targetX = pasteX + sx;
                    int targetY = pasteY + sy;
                    
                    if (targetX >= 0 && targetX < result.width && targetY >= 0 && targetY < result.height)
                    {
                        resultPixels[targetY * result.width + targetX] = smallPixels[sy * maskSmall.width + sx];
                    }
                }
            }
            
            result.LoadRawTextureData(resultPixels);
            result.Apply();
            
            return result;
        }

        /// <summary>
        /// Create small mask cropped to face region (matching Python mask_small)
        /// </summary>
        private static Texture2D CreateSmallMask(Texture2D maskTexture, Vector4 faceBbox, Rect cropBox)
        {
            // Python: mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))
            float x = faceBbox.x;
            float y = faceBbox.y;
            float x1 = faceBbox.z;
            float y1 = faceBbox.w;
            float x_s = cropBox.x;
            float y_s = cropBox.y;
            
            Rect cropRect = new Rect(
                x - x_s,
                y - y_s,
                x1 - x - (x - x_s),  // width = x1 - x
                y1 - y - (y - y_s)   // height = y1 - y
            );
            
            return CropImage(maskTexture, cropRect);
        }

        /// <summary>
        /// Create full mask by pasting small mask into blank canvas (matching Python mask_full)
        /// OPTIMIZED: Uses unsafe pointers, parallelization, and memcpy for maximum performance
        /// </summary>
        public static unsafe Texture2D CreateFullMask(Texture2D originalMask, Texture2D smallMask, Vector4 faceBbox, Rect cropBox)
        {
            // Python: mask_image = Image.new('L', ori_shape, 0)
            // Python: mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))
            
            int width = originalMask.width;
            int height = originalMask.height;
            
            // Create blank mask with RGB24 format for efficiency
            var fullMask = new Texture2D(width, height, TextureFormat.RGB24, false);
            var fullPixelData = fullMask.GetPixelData<byte>(0);
            var smallPixelData = smallMask.GetPixelData<byte>(0);
            
            // Get unsafe pointers for direct memory operations
            byte* fullPtr = (byte*)fullPixelData.GetUnsafePtr();
            byte* smallPtr = (byte*)smallPixelData.GetUnsafeReadOnlyPtr();
            
            // Initialize to black (parallel clear for large textures)
            int totalBytes = width * height * 3; // RGB24: 3 bytes per pixel
            System.Threading.Tasks.Parallel.For(0, height, y =>
            {
                byte* rowPtr = fullPtr + y * width * 3;
                UnsafeUtility.MemClear(rowPtr, width * 3);
            });
            
            // Calculate paste position (matching Python coordinates)
            float x = faceBbox.x;
            float y = faceBbox.y;
            float x_s = cropBox.x;
            float y_s = cropBox.y;
            
            int pasteX = Mathf.RoundToInt(x - x_s);
            int pasteY = Mathf.RoundToInt(y - y_s);
            
            // Calculate valid paste region to avoid bounds checking in inner loop
            int startX = Mathf.Max(0, pasteX);
            int endX = Mathf.Min(width, pasteX + smallMask.width);
            int startY = Mathf.Max(0, pasteY);
            int endY = Mathf.Min(height, pasteY + smallMask.height);
            
            // Parallel paste operation - process rows concurrently with bulk copy optimization
            System.Threading.Tasks.Parallel.For(startY, endY, targetY =>
            {
                int sourceY = targetY - pasteY;
                if (sourceY >= 0 && sourceY < smallMask.height)
                {
                    byte* targetRowPtr = fullPtr + targetY * width * 3;
                    byte* sourceRowPtr = smallPtr + sourceY * smallMask.width * 3;
                    
                    // Check if we can do a full row copy (when pasting starts at X=0 and covers full width)
                    if (pasteX == 0 && startX == 0 && endX == width && smallMask.width == width)
                    {
                        // Bulk copy entire row using native memcpy (fastest path)
                        UnsafeUtility.MemCpy(targetRowPtr, sourceRowPtr, smallMask.width * 3);
                    }
                    else
                    {
                        // Partial row copy - optimized pixel-by-pixel with pointer arithmetic
                        for (int targetX = startX; targetX < endX; targetX++)
                        {
                            int sourceX = targetX - pasteX;
                            
                            // Copy RGB values from small mask (grayscale: R=G=B)
                            byte* targetPixel = targetRowPtr + targetX * 3;
                            byte* sourcePixel = sourceRowPtr + sourceX * 3;
                            
                            // Use red channel as grayscale value for all RGB components
                            byte grayValue = sourcePixel[0]; // Red channel
                            targetPixel[0] = grayValue; // R
                            targetPixel[1] = grayValue; // G
                            targetPixel[2] = grayValue; // B
                        }
                    }
                }
            });
            
            fullMask.Apply();
            return fullMask;
        }

        /// <summary>
        /// Apply upper boundary ratio to preserve upper face area (matching Python)
        /// OPTIMIZED: Uses unsafe pointers, parallelization, and memcpy for maximum performance
        /// </summary>
        public static unsafe Texture2D ApplyUpperBoundaryRatio(Texture2D mask, float upperBoundaryRatio)
        {
            int width = mask.width;
            int height = mask.height;
            
            // Create result texture with RGB24 format for efficiency
            var result = new Texture2D(width, height, TextureFormat.RGB24, false);
            var maskPixelData = mask.GetPixelData<byte>(0);
            var resultPixelData = result.GetPixelData<byte>(0);
            
            // Get unsafe pointers for direct memory operations
            byte* maskPtr = (byte*)maskPixelData.GetUnsafeReadOnlyPtr();
            byte* resultPtr = (byte*)resultPixelData.GetUnsafePtr();
            
            // Calculate boundary line (matching Python upper_boundary_ratio logic)
            // In Unity: Y=0 is bottom, Y=height-1 is top
            // upperBoundaryRatio=0.5 means preserve top 50% of face (upper half)
            int boundaryY = Mathf.RoundToInt(height * (1.0f - upperBoundaryRatio));
            
            // Parallel processing of rows for maximum performance
            System.Threading.Tasks.Parallel.For(0, height, y =>
            {
                byte* maskRowPtr = maskPtr + y * width * 3;
                byte* resultRowPtr = resultPtr + y * width * 3;
                
                // In Unity coordinates: higher Y = upper part of image
                if (y >= boundaryY) // Upper part of face (preserve original)
                {
                    // Zero out entire row using fast memclear (preserve original face - eyes, nose, forehead)
                    UnsafeUtility.MemClear(resultRowPtr, width * 3);
                }
                else // Lower part of face (talking area - mouth, chin)
                {
                    // Copy entire row from mask to result using fast memcpy (allow blending)
                    UnsafeUtility.MemCpy(resultRowPtr, maskRowPtr, width * 3);
                }
            });
            
            result.Apply();
            return result;
        }

        /// <summary>
        /// Composite images using mask-based blending (matching Python PIL paste logic)
        /// </summary>
        private static Texture2D CompositeImages(
            Texture2D originalImage,
            Texture2D generatedFace,
            Texture2D faceLarge,
            Vector4 faceBbox,
            Rect cropBox,
            Texture2D blurredMask)
        {
            // Step 1: Paste generated face into face_large at relative position
            var compositeFaceLarge = PasteFaceIntoLarge(faceLarge, generatedFace, faceBbox, cropBox);
            
            // Step 2: Paste composite face_large back into original image using mask
            var result = PasteWithMask(originalImage, compositeFaceLarge, cropBox, blurredMask);
            
            return result;
        }

        /// <summary>
        /// Paste generated face into the large face region
        /// OPTIMIZED: Uses unsafe pointers, parallelization, and memcpy for maximum performance
        /// </summary>
        private static unsafe Texture2D PasteFaceIntoLarge(
            Texture2D faceLarge,
            Texture2D generatedFace,
            Vector4 faceBbox,
            Rect cropBox)
        {
            // Calculate relative position within face_large (image coordinates) - use exact integer calculation like Python
            int relativeX = (int)(faceBbox.x - cropBox.x);
            int relativeY = (int)(faceBbox.y - cropBox.y);
            
            // CRITICAL: Convert from image coordinates (Y=0 at top) to Unity coordinates (Y=0 at bottom)
            int unityRelativeY = faceLarge.height - relativeY - generatedFace.height;
            
            // Create result texture with RGB24 format for efficiency
            var result = new Texture2D(faceLarge.width, faceLarge.height, TextureFormat.RGB24, false);
            var faceLargePixelData = faceLarge.GetPixelData<byte>(0);
            var generatedFacePixelData = generatedFace.GetPixelData<byte>(0);
            var resultPixelData = result.GetPixelData<byte>(0);
            
            // Get unsafe pointers for direct memory operations
            byte* faceLargePtr = (byte*)faceLargePixelData.GetUnsafeReadOnlyPtr();
            byte* generatedFacePtr = (byte*)generatedFacePixelData.GetUnsafeReadOnlyPtr();
            byte* resultPtr = (byte*)resultPixelData.GetUnsafePtr();
            
            int resultWidth = result.width;
            int resultHeight = result.height;
            int faceWidth = generatedFace.width;
            int faceHeight = generatedFace.height;
            
            // First, copy the entire faceLarge to result using parallel memcpy
            System.Threading.Tasks.Parallel.For(0, resultHeight, y =>
            {
                byte* sourceRowPtr = faceLargePtr + y * resultWidth * 3;
                byte* targetRowPtr = resultPtr + y * resultWidth * 3;
                UnsafeUtility.MemCpy(targetRowPtr, sourceRowPtr, resultWidth * 3);
            });
            
            // Calculate valid paste region to avoid bounds checking in inner loop
            int startX = Mathf.Max(0, relativeX);
            int endX = Mathf.Min(resultWidth, relativeX + faceWidth);
            int startY = Mathf.Max(0, unityRelativeY);
            int endY = Mathf.Min(resultHeight, unityRelativeY + faceHeight);
            
            // Parallel paste operation - process rows of the generated face
            System.Threading.Tasks.Parallel.For(startY, endY, targetY =>
            {
                int sourceY = targetY - unityRelativeY;
                if (sourceY >= 0 && sourceY < faceHeight)
                {
                    byte* targetRowPtr = resultPtr + targetY * resultWidth * 3;
                    byte* sourceRowPtr = generatedFacePtr + sourceY * faceWidth * 3;
                    
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
            
            result.Apply();
            return result;
        }

        /// <summary>
        /// Paste image with mask-based alpha blending
        /// </summary>
        private static Texture2D PasteWithMask(
            Texture2D background,
            Texture2D foreground,
            Rect cropBox,
            Texture2D mask)
        {
            var result = new Texture2D(background.width, background.height, background.format, false);
            var bgPixelsOriginal = background.GetPixels();
            result.SetPixels(bgPixelsOriginal);
            result.Apply();
            
            var bgPixels = result.GetPixels();
            var fgPixels = foreground.GetPixels();
            var maskPixels = mask.GetRawTextureData<byte>();
            
            int cropX = Mathf.RoundToInt(cropBox.x);
            int cropY = Mathf.RoundToInt(cropBox.y);
            
            for (int y = 0; y < foreground.height && y + cropY < result.height; y++)
            {
                for (int x = 0; x < foreground.width && x + cropX < result.width; x++)
                {
                    if (x + cropX >= 0 && y + cropY >= 0)
                    {
                        int bgIndex = (y + cropY) * result.width + (x + cropX);
                        int fgIndex = y * foreground.width + x;
                        int maskIndex = y * mask.width + x;
                        
                        if (bgIndex < bgPixels.Length && fgIndex < fgPixels.Length && maskIndex < maskPixels.Length)
                        {
                            float alpha = maskPixels[maskIndex] / 255f;
                            
                            // Alpha blending
                            Color bgColor = bgPixels[bgIndex];
                            Color fgColor = fgPixels[fgIndex];
                            
                            bgPixels[bgIndex] = Color.Lerp(bgColor, fgColor, alpha);
                        }
                    }
                }
            }
            
            result.SetPixels(bgPixels);
            result.Apply();
            
            return result;
        }
        
        /// <summary>
        /// Apply segmentation mask to blend face with original image using precomputed masks
        /// OPTIMIZED: Uses precomputed masks for maximum performance
        /// </summary>
        public static Texture2D ApplySegmentationMaskWithPrecomputedMasks(Texture2D originalImage, Texture2D faceTexture, 
            Vector4 faceBbox, Vector4 cropBox, Texture2D precomputedBlurredMask, Texture2D precomputedFaceLarge)
        {
            try
            {
                // Step 1: Use precomputed face_large and paste the resized face into it (matching Python exactly)
                // Python: face_large = body.crop(crop_box); face_large.paste(face, (x-x_s, y-y_s))
                // Note: faceLarge is already precomputed, no need to crop again
                
                // Resize face texture to match face bbox dimensions (use exact integer calculation like Python)
                int faceWidth = (int)(faceBbox.z - faceBbox.x);
                int faceHeight = (int)(faceBbox.w - faceBbox.y);
                var resizedFace = TextureUtils.ResizeTexture(faceTexture, faceWidth, faceHeight);

                // Paste the resized face into precomputed face_large at relative position (matching Python)
                // Python: face_large.paste(face, (x-x_s, y-y_s))
                var faceLargeWithFace = PasteFaceIntoLarge(precomputedFaceLarge, resizedFace, faceBbox, new Rect(cropBox.x, cropBox.y, cropBox.z - cropBox.x, cropBox.w - cropBox.y));
                // Step 2: Composite images using the precomputed blurred mask (matching Python alpha blending)
                // Python: body.paste(face_large, crop_box[:2], mask_image)
                var result = CompositeWithMask(originalImage, faceLargeWithFace, faceBbox, precomputedBlurredMask, cropBox);

                // Cleanup intermediate textures (don't destroy precomputed faceLarge as it's cached)
                UnityEngine.Object.Destroy(resizedFace);
                UnityEngine.Object.Destroy(faceLargeWithFace);
                
                return result;
            }
            catch (Exception e)
            {
                Logger.LogError($"[ImageBlendingHelper] Precomputed mask blending failed: {e.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Apply Gaussian blur to mask for smooth blending (matching Python)
        /// </summary>
        public static Texture2D ApplyGaussianBlurToMask(Texture2D mask)
        {
            // Calculate blur kernel size based on mask dimensions (matching Python)
            float blurFactor = 0.08f; // jaw mode blur factor from Python
            int kernelSize = Mathf.RoundToInt(blurFactor * mask.width / 2) * 2 + 1;
            kernelSize = Mathf.Max(kernelSize, 15); // Minimum kernel size
            
            return TextureUtils.ApplySimpleGaussianBlur(mask, kernelSize);
        }
        
        /// <summary>
        /// Composite images using blurred mask (matching Python exactly)
        /// OPTIMIZED: Uses unsafe pointers, parallelization, and optimized alpha blending
        /// Python: body.paste(face_large, crop_box[:2], mask_image)
        /// </summary>
        private static unsafe Texture2D CompositeWithMask(Texture2D originalImage, Texture2D faceLarge, 
            Vector4 faceBbox, Texture2D blurredMask, Vector4 cropBox)
        {
            int resultWidth = originalImage.width;
            int resultHeight = originalImage.height;
            int blendWidth = faceLarge.width;
            int blendHeight = faceLarge.height;
            
            // Create result texture with RGB24 format for efficiency
            var result = new Texture2D(resultWidth, resultHeight, TextureFormat.RGB24, false);
            var originalPixelData = originalImage.GetPixelData<byte>(0);
            var faceLargePixelData = faceLarge.GetPixelData<byte>(0);
            var maskPixelData = blurredMask.GetPixelData<byte>(0);
            var resultPixelData = result.GetPixelData<byte>(0);
            
            // Get unsafe pointers for direct memory operations
            byte* originalPtr = (byte*)originalPixelData.GetUnsafeReadOnlyPtr();
            byte* faceLargePtr = (byte*)faceLargePixelData.GetUnsafeReadOnlyPtr();
            byte* maskPtr = (byte*)maskPixelData.GetUnsafeReadOnlyPtr();
            byte* resultPtr = (byte*)resultPixelData.GetUnsafePtr();
            
            // Calculate paste position (matching Python: crop_box[:2])
            int pasteX = (int)cropBox.x;
            int pasteY = (int)cropBox.y;
            
            // Convert image coordinates to Unity coordinates (flip Y axis)
            int unityPasteY = resultHeight - pasteY - blendHeight;
            
            // First, copy the entire original image to result using parallel memcpy
            System.Threading.Tasks.Parallel.For(0, resultHeight, y =>
            {
                byte* sourceRowPtr = originalPtr + y * resultWidth * 3;
                byte* targetRowPtr = resultPtr + y * resultWidth * 3;
                UnsafeUtility.MemCpy(targetRowPtr, sourceRowPtr, resultWidth * 3);
            });
            
            // Calculate valid blend region to avoid bounds checking in inner loop
            int startX = Mathf.Max(0, pasteX);
            int endX = Mathf.Min(resultWidth, pasteX + blendWidth);
            int startY = Mathf.Max(0, unityPasteY);
            int endY = Mathf.Min(resultHeight, unityPasteY + blendHeight);
            
            // Parallel alpha blending - process rows concurrently
            System.Threading.Tasks.Parallel.For(startY, endY, targetY =>
            {
                int sourceY = targetY - unityPasteY;
                if (sourceY >= 0 && sourceY < blendHeight)
                {
                    byte* resultRowPtr = resultPtr + targetY * resultWidth * 3;
                    byte* faceLargeRowPtr = faceLargePtr + sourceY * blendWidth * 3;
                    byte* maskRowPtr = maskPtr + sourceY * blendWidth * 3;
                    
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
            
            result.Apply();
            return result;
        }
    }
} 