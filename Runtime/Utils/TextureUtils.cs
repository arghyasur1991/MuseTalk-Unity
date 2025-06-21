using System;
using UnityEngine;
using Unity.Collections.LowLevel.Unsafe;

namespace LiveTalk.Utils
{
    using Core;
    /// <summary>
    /// Utility functions for texture processing in MuseTalk
    /// </summary>
    public static class TextureUtils
    {
        /// <summary>
        /// Check if texture format is compressed
        /// </summary>
        private static bool IsCompressedFormat(TextureFormat format)
        {
            return format == TextureFormat.DXT1 ||
                   format == TextureFormat.DXT5 ||
                   format == TextureFormat.BC4 ||
                   format == TextureFormat.BC5 ||
                   format == TextureFormat.BC6H ||
                   format == TextureFormat.BC7 ||
                   format == TextureFormat.ETC_RGB4 ||
                   format == TextureFormat.ETC2_RGBA8 ||
                   format == TextureFormat.ASTC_4x4 ||
                   format == TextureFormat.ASTC_5x5 ||
                   format == TextureFormat.ASTC_6x6 ||
                   format == TextureFormat.ASTC_8x8 ||
                   format == TextureFormat.ASTC_10x10 ||
                   format == TextureFormat.ASTC_12x12;
        }
        
        /// <summary>
        /// Create a readable copy of texture using RenderTexture
        /// </summary>
        public static Texture2D MakeTextureReadable(Texture2D source)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            
            // Create RenderTexture
            RenderTexture renderTexture = RenderTexture.GetTemporary(
                source.width, 
                source.height, 
                0, 
                RenderTextureFormat.ARGB32
            );
            
            RenderTexture previousActive = RenderTexture.active;
            RenderTexture.active = renderTexture;
            
            try
            {
                // Blit source texture to RenderTexture
                Graphics.Blit(source, renderTexture);
                
                // Create new readable texture
                Texture2D readableTexture = new(source.width, source.height, TextureFormat.RGB24, false);
                readableTexture.ReadPixels(new Rect(0, 0, source.width, source.height), 0, 0);
                readableTexture.Apply();
                
                return readableTexture;
            }
            finally
            {
                // Clean up
                RenderTexture.active = previousActive;
                RenderTexture.ReleaseTemporary(renderTexture);
            }
        }
        
        /// <summary>
        /// Resize texture to specified dimensions
        /// </summary>
        public static Texture2D ResizeTexture(Texture2D source, int targetWidth, int targetHeight)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
                
            // Create RenderTexture for scaling
            RenderTexture renderTexture = RenderTexture.GetTemporary(targetWidth, targetHeight, 0, RenderTextureFormat.ARGB32);
            RenderTexture previousActive = RenderTexture.active;
            RenderTexture.active = renderTexture;
            
            try
            {
                // Scale the source texture
                Graphics.Blit(source, renderTexture);
                
                // Create new texture from RenderTexture
                Texture2D resizedTexture = new Texture2D(targetWidth, targetHeight, TextureFormat.RGB24, false);
                resizedTexture.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
                resizedTexture.Apply();
                
                return resizedTexture;
            }
            finally
            {
                // Clean up
                RenderTexture.active = previousActive;
                RenderTexture.ReleaseTemporary(renderTexture);
            }
        }

        public static Texture2D ConvertTexture2DToRGB24(Texture2D texture)
        {
            if (texture.format != TextureFormat.RGB24)
            {
                var convertedTexture = new Texture2D(texture.width, texture.height, TextureFormat.RGB24, false)
                {
                    name = texture.name
                };
                convertedTexture.SetPixels(texture.GetPixels());
                convertedTexture.Apply();
                return convertedTexture;
            }
            return texture;
        }

        /// <summary>
        /// Convert Texture2D to byte array (RGB24 format)
        /// </summary>
        public static unsafe Frame Texture2DToFrame(Texture2D img)
        {
            int h = img.height;
            int w = img.width;
            int rowBytes = w * 3; // RGB24 = 3 bytes per pixel
            
            // Get initial image data directly from texture (assumes RGB24 format)
            var pixelData = img.GetPixelData<byte>(0);
            var imageData = new byte[pixelData.Length];
            
            byte* srcPtr = (byte*)pixelData.GetUnsafeReadOnlyPtr();
            
            fixed (byte* dstPtr = imageData)
            {
                byte* srcPtrLocal = srcPtr;
                byte* dstPtrLocal = dstPtr;
                
                System.Threading.Tasks.Parallel.For(0, h, y =>
                {
                    byte* srcRowPtr = srcPtrLocal + (h - 1 - y) * rowBytes; // Source row (flipped)
                    byte* dstRowPtr = dstPtrLocal + y * rowBytes;            // Destination row
                    Buffer.MemoryCopy(srcRowPtr, dstRowPtr, rowBytes, rowBytes);
                });
            }

            return new Frame(imageData, w, h);
        }
        
        /// <summary>
        /// Convert RGB24 byte array back to Texture2D using unsafe pointers and parallelization
        /// </summary>
        public static unsafe Texture2D FrameToTexture2D(Frame frame)
        {
            var texture = new Texture2D(frame.width, frame.height, TextureFormat.RGB24, false);
            
            var pixelData = texture.GetPixelData<byte>(0);
            byte* texturePtr = (byte*)pixelData.GetUnsafePtr();
            
            // OPTIMIZED: Process with unsafe pointers and parallelization
            fixed (byte* imagePtrFixed = frame.data)
            {
                // Capture pointer in local variable to avoid lambda closure issues
                byte* imagePtrLocal = imagePtrFixed;
                
                System.Threading.Tasks.Parallel.For(0, frame.height, y =>
                {
                    // Calculate Unity texture coordinate (bottom-left origin) from image coordinate (top-left origin)
                    int unityY = frame.height - 1 - y; // Flip Y coordinate for Unity coordinate system
                    
                    // Calculate row pointers using direct pointer arithmetic
                    byte* srcRowPtr = imagePtrLocal + y * frame.width * 3;        // Source row (top-left origin)
                    byte* dstRowPtr = texturePtr + unityY * frame.width * 3;      // Destination row (bottom-left origin)
                    
                    int rowBytes = frame.width * 3; // RGB24 = 3 bytes per pixel
                    Buffer.MemoryCopy(srcRowPtr, dstRowPtr, rowBytes, rowBytes);
                });
            }
            
            // Apply changes to texture (no need for SetPixels since we wrote directly to pixel data)
            texture.Apply();
            return texture;
        }

        /// <summary>
        /// Generate a content-based hash for a texture
        /// </summary>
        public static string GenerateTextureHash(Texture2D texture)
        {
            // Use texture properties and a sample of pixels for hashing
            // This is faster than hashing all pixels but still provides good uniqueness
            unchecked
            {
                int hash = texture.width.GetHashCode();
                hash = hash * 31 + texture.height.GetHashCode();
                hash = hash * 31 + texture.format.GetHashCode();
                
                // Sample a few pixels for content-based hashing (much faster than full texture)
                var pixels = texture.GetPixels(0, 0, Math.Min(32, texture.width), Math.Min(32, texture.height));
                for (int i = 0; i < Math.Min(100, pixels.Length); i += 10) // Sample every 10th pixel
                {
                    hash = hash * 31 + pixels[i].GetHashCode();
                }
                
                return hash.ToString("X8");
            }
        }
    }
}