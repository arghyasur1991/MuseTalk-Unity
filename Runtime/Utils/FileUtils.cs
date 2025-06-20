using System.Collections.Generic;
using System;
using UnityEngine;
using System.Linq;

namespace MuseTalk.Utils
{
    public static class FileUtils
    {
        /// <summary>
        /// Get driving frame file paths for counting and loading
        /// </summary>
        public static string[] GetFrameFiles(string framesFolderPath, int maxFrames = -1)
        {
            var supportedExtensions = new string[] { ".png", ".jpg", ".jpeg" };
            string fullFolderPath = System.IO.Path.Combine(Application.streamingAssetsPath, framesFolderPath);
            
            if (!System.IO.Directory.Exists(fullFolderPath))
            {
                return new string[0];
            }

            var allFiles = new List<string>();
            foreach (string extension in supportedExtensions)
            {
                string[] files = System.IO.Directory.GetFiles(fullFolderPath, "*" + extension, System.IO.SearchOption.TopDirectoryOnly);
                allFiles.AddRange(files);
            }

            // Sort files by name for consistent ordering
            allFiles.Sort((a, b) => string.Compare(System.IO.Path.GetFileNameWithoutExtension(a), 
                                                  System.IO.Path.GetFileNameWithoutExtension(b), 
                                                  System.StringComparison.Ordinal));

            if (maxFrames > 0)
            {
                allFiles = allFiles.Take(maxFrames).ToList();
            }

            return allFiles.ToArray();
        }
        public static List<Texture2D> LoadFramesFromFolder(string drivingFramesFolderPath)
        {
            var supportedExtensions = new string[] { ".png", ".jpg", ".jpeg" };
            // First try to load from folder if specified
            if (!string.IsNullOrEmpty(drivingFramesFolderPath))
            {
                string fullFolderPath = System.IO.Path.Combine(Application.streamingAssetsPath, drivingFramesFolderPath);
                
                if (System.IO.Directory.Exists(fullFolderPath))
                {
                    var framesList = new List<Texture2D>();
                    
                    // Get all image files from folder
                    foreach (string extension in supportedExtensions)
                    {
                        string[] files = System.IO.Directory.GetFiles(fullFolderPath, "*" + extension, System.IO.SearchOption.TopDirectoryOnly);
                        
                        foreach (string filePath in files)
                        {
                            try
                            {
                                byte[] fileData = System.IO.File.ReadAllBytes(filePath);
                                Texture2D texture = new(2, 2);                                
                                if (texture.LoadImage(fileData))
                                {
                                    texture.name = System.IO.Path.GetFileNameWithoutExtension(filePath);
                                    framesList.Add(TextureUtils.ConvertTexture2DToRGB24(texture));
                                }
                                else
                                {
                                    Debug.LogWarning($"Failed to load image: {filePath}");
                                    UnityEngine.Object.DestroyImmediate(texture);
                                }
                            }
                            catch (Exception e)
                            {
                                Debug.LogError($"Error loading driving frame {filePath}: {e.Message}");
                            }
                        }
                    }
                    
                    if (framesList.Count > 0)
                    {
                        // Sort frames by name (handles numbered sequences like 00000000, 00000001, etc.)
                        framesList.Sort((a, b) => string.Compare(a.name, b.name, System.StringComparison.Ordinal));
                        
                        return framesList;
                    }
                    else
                    {
                        Debug.LogWarning($"No valid images found in driving frames folder: {fullFolderPath}");
                    }
                }
                else
                {
                    Debug.LogWarning($"Driving frames folder not found: {fullFolderPath}");
                }
            }
            return null;
        }
    }
}