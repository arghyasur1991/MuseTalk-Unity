using System;
using System.IO;
using UnityEngine;

namespace LiveTalk.Utils
{
    /// <summary>
    /// Enhanced debug logger with file dumping capabilities
    /// </summary>
    internal class DebugLogger
    {
        private static bool _enableLogging = true;
        private static bool _enableFileDebug = false;
        private static readonly string _debugOutputPath;
        
        static DebugLogger()
        {
            // Set debug output path to Resources folder for easy access
            _debugOutputPath = Path.Combine(Application.streamingAssetsPath, "MuseTalkDebug");
            
            // Create debug directory if it doesn't exist
            if (_enableFileDebug && !Directory.Exists(_debugOutputPath))
            {
                Directory.CreateDirectory(_debugOutputPath);
                Debug.Log($"[DebugLogger] Created debug output directory: {_debugOutputPath}");
            }
        }
        
        public static bool EnableLogging
        {
            get => _enableLogging;
            set => _enableLogging = value;
        }
        
        public static bool EnableFileDebug
        {
            get => _enableFileDebug;
            set => _enableFileDebug = value;
        }
        
        public static string DebugOutputPath => _debugOutputPath;
        
        public void Log(string message)
        {
            if (_enableLogging)
            {
                Debug.Log(message);
            }
        }
        
        public void LogWarning(string message)
        {
            if (_enableLogging)
            {
                Debug.LogWarning(message);
            }
        }
        
        public void LogError(string message)
        {
            if (_enableLogging)
            {
                Debug.LogError(message);
            }
        }
        
        /// <summary>
        /// Dump texture to debug folder for inspection
        /// </summary>
        public static void DumpTexture(Texture2D texture, string filename, string subfolder = "")
        {
            if (!_enableFileDebug || texture == null) return;
            
            try
            {
                string folderPath = string.IsNullOrEmpty(subfolder) 
                    ? _debugOutputPath 
                    : Path.Combine(_debugOutputPath, subfolder);
                    
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }
                
                string filePath = Path.Combine(folderPath, $"{filename}.png");
                
                // Ensure texture is readable
                Texture2D readableTexture = texture;
                if (!texture.isReadable)
                {
                    readableTexture = TextureUtils.MakeTextureReadable(texture);
                }
                
                byte[] pngData = readableTexture.EncodeToPNG();
                File.WriteAllBytes(filePath, pngData);
                
                Debug.Log($"[DebugLogger] Dumped texture: {filePath} ({texture.width}×{texture.height})");
                
                // Clean up if we created a copy
                if (readableTexture != texture)
                {
                    UnityEngine.Object.DestroyImmediate(readableTexture);
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[DebugLogger] Failed to dump texture {filename}: {e.Message}");
            }
        }
        
        /// <summary>
        /// Dump float array data to text file for inspection
        /// </summary>
        public static void DumpFloatArray(float[] data, string filename, string subfolder = "", int maxElements = 100)
        {
            if (!_enableFileDebug || data == null) return;
            
            try
            {
                string folderPath = string.IsNullOrEmpty(subfolder) 
                    ? _debugOutputPath 
                    : Path.Combine(_debugOutputPath, subfolder);
                    
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }
                
                string filePath = Path.Combine(folderPath, $"{filename}.txt");
                
                using (var writer = new StreamWriter(filePath))
                {
                    writer.WriteLine($"Array Length: {data.Length}");
                    writer.WriteLine($"Data Type: float[]");
                    writer.WriteLine($"Sample (first {Math.Min(maxElements, data.Length)} elements):");
                    writer.WriteLine();
                    
                    int elementsToShow = Math.Min(maxElements, data.Length);
                    for (int i = 0; i < elementsToShow; i++)
                    {
                        writer.WriteLine($"[{i}] = {data[i]:F6}");
                    }
                    
                    if (data.Length > maxElements)
                    {
                        writer.WriteLine($"... and {data.Length - maxElements} more elements");
                    }
                    
                    // Statistics
                    float min = float.MaxValue, max = float.MinValue, sum = 0f;
                    foreach (float value in data)
                    {
                        if (value < min) min = value;
                        if (value > max) max = value;
                        sum += value;
                    }
                    
                    writer.WriteLine();
                    writer.WriteLine("Statistics:");
                    writer.WriteLine($"Min: {min:F6}");
                    writer.WriteLine($"Max: {max:F6}");
                    writer.WriteLine($"Mean: {(sum / data.Length):F6}");
                }
                
                Debug.Log($"[DebugLogger] Dumped float array: {filePath} ({data.Length} elements)");
            }
            catch (Exception e)
            {
                Debug.LogError($"[DebugLogger] Failed to dump float array {filename}: {e.Message}");
            }
        }
        
        /// <summary>
        /// Dump tensor shape and sample data
        /// </summary>
        public static void DumpTensorInfo(Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> tensor, string filename, string subfolder = "")
        {
            if (!_enableFileDebug || tensor == null) return;
            
            try
            {
                string folderPath = string.IsNullOrEmpty(subfolder) 
                    ? _debugOutputPath 
                    : Path.Combine(_debugOutputPath, subfolder);
                    
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }
                
                string filePath = Path.Combine(folderPath, $"{filename}_tensor.txt");
                var tensorData = new float[tensor.Length];
                for (int i = 0; i < tensor.Length; i++)
                {
                    tensorData[i] = tensor.GetValue(i);
                }
                var dimensions = tensor.Dimensions.ToArray();
                
                using (var writer = new StreamWriter(filePath))
                {
                    writer.WriteLine($"Tensor Shape: [{string.Join(", ", dimensions)}]");
                    writer.WriteLine($"Total Elements: {tensorData.Length}");
                    writer.WriteLine($"Data Type: float");
                    writer.WriteLine();
                    
                    // Sample data (first 50 elements)
                    writer.WriteLine("Sample Data (first 50 elements):");
                    int sampleCount = Math.Min(50, tensorData.Length);
                    for (int i = 0; i < sampleCount; i++)
                    {
                        writer.WriteLine($"[{i}] = {tensorData[i]:F6}");
                    }
                    
                    // Statistics
                    float min = float.MaxValue, max = float.MinValue, sum = 0f;
                    foreach (float value in tensorData)
                    {
                        if (value < min) min = value;
                        if (value > max) max = value;
                        sum += value;
                    }
                    
                    writer.WriteLine();
                    writer.WriteLine("Statistics:");
                    writer.WriteLine($"Min: {min:F6}");
                    writer.WriteLine($"Max: {max:F6}");
                    writer.WriteLine($"Mean: {(sum / tensorData.Length):F6}");
                }
                
                Debug.Log($"[DebugLogger] Dumped tensor info: {filePath}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[DebugLogger] Failed to dump tensor info {filename}: {e.Message}");
            }
        }
        
        /// <summary>
        /// Dump face detection debug with coordinate system info
        /// </summary>
        public static void DumpFaceDetectionResults(Vector4[] bboxes, string filename, string subfolder)
        {
            var debugPath = Path.Combine(DebugOutputPath, subfolder);
            Directory.CreateDirectory(debugPath);
            
            var filePath = Path.Combine(debugPath, $"{filename}_bboxes.txt");
            
            using (var writer = new StreamWriter(filePath))
            {
                writer.WriteLine("=== FACE DETECTION RESULTS ===");
                writer.WriteLine($"Detection Count: {bboxes.Length}");
                writer.WriteLine($"Coordinate System: Top-left origin (Python/OpenCV style)");
                writer.WriteLine("");
                
                for (int i = 0; i < bboxes.Length; i++)
                {
                    var bbox = bboxes[i];
                    if (bbox == Vector4.zero)
                    {
                        writer.WriteLine($"Face {i}: NO FACE DETECTED");
                    }
                    else
                    {
                        writer.WriteLine($"Face {i}:");
                        writer.WriteLine($"  BBox (x1,y1,x2,y2): [{bbox.x:F1}, {bbox.y:F1}, {bbox.z:F1}, {bbox.w:F1}]");
                        writer.WriteLine($"  Width: {bbox.z - bbox.x:F1}");
                        writer.WriteLine($"  Height: {bbox.w - bbox.y:F1}");
                        writer.WriteLine($"  Unity Crop Y: {bbox.y:F1} -> will convert to bottom-left origin");
                        writer.WriteLine("");
                    }
                }
                
                writer.WriteLine("NOTE: Y coordinates will be converted from top-left to bottom-left origin during cropping");
            }
            
            if (EnableLogging)
                Debug.Log($"[DebugLogger] Saved face detection results: {filePath}");
        }
        
        /// <summary>
        /// Clear debug output directory
        /// </summary>
        public static void ClearDebugOutput()
        {
            if (!_enableFileDebug) return;
            
            try
            {
                if (Directory.Exists(_debugOutputPath))
                {
                    Directory.Delete(_debugOutputPath, true);
                    Directory.CreateDirectory(_debugOutputPath);
                    Debug.Log("[DebugLogger] Cleared debug output directory");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[DebugLogger] Failed to clear debug output: {e.Message}");
            }
        }
    }
} 