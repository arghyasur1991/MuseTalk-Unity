using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using UnityEngine;

namespace MuseTalk.Utils
{
    using Models;
    /// <summary>
    /// Utility functions for ONNX model loading and configuration in MuseTalk
    /// </summary>
    public static class ModelUtils
    {
        /// <summary>
        /// Create optimized session options for ONNX Runtime
        /// OPTIMIZED: Enable all performance optimizations
        /// </summary>
        private static SessionOptions CreateSessionOptions(MuseTalkConfig config)
        {
            var options = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                ExecutionMode = ExecutionMode.ORT_PARALLEL,

                EnableMemoryPattern = true,
                EnableCpuMemArena = true,
                InterOpNumThreads = Environment.ProcessorCount,
                IntraOpNumThreads = Environment.ProcessorCount
            };

            if (config.UseINT8)
            {
                Debug.Log("[MuseTalkInference] Enabling INT8 quantization optimizations");
                // INT8 models work best with all optimizations enabled
                options.AddSessionConfigEntry("session.enable_memory_arena_shrinkage", "cpu:0;");
            }
            
            options.AddSessionConfigEntry("session.disable_prepacking", "0"); // Enable weight prepacking
            options.AddSessionConfigEntry("session.use_env_allocators", "1"); // Use environment allocators
            
            options.AddSessionConfigEntry("session.intra_op_param", ""); // Let ORT auto-tune
            options.AddSessionConfigEntry("session.inter_op_param", ""); // Let ORT auto-tune
            
            try
            {
                // Try CUDA first (NVIDIA GPUs) with optimized settings
                var cudaOptions = new Dictionary<string, string>
                {
                    ["device_id"] = "0",
                    ["arena_extend_strategy"] = "kSameAsRequested", // Optimize memory allocation
                    ["gpu_mem_limit"] = "0", // Use all available GPU memory
                    ["cudnn_conv_algo_search"] = "EXHAUSTIVE", // Find best conv algorithms
                    ["do_copy_in_default_stream"] = "1", // Optimize memory transfers
                };
                
                options.AppendExecutionProvider("CUDAExecutionProvider", cudaOptions);
                Debug.Log("[MuseTalkInference] CUDA GPU provider enabled with optimizations");
            }
            catch (Exception)
            {
                try
                {
                    // Try DirectML (Windows GPU acceleration)
                    options.AppendExecutionProvider_DML(0);
                    Debug.Log("[MuseTalkInference] DirectML GPU provider enabled");
                }
                catch (Exception)
                {
                    Debug.Log("[MuseTalkInference] Using CPU execution provider (GPU not available)");
                }
            }
            
            return options;
        }

        public static InferenceSession LoadModel(MuseTalkConfig config, string modelName)
        {
            string modelPath = GetModelPath(config, modelName);
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"{modelName} model not found: {modelPath}");
            var sessionOptions = CreateSessionOptions(config);
            if (
                modelName == "det_10g" || 
                modelName == "1k3d68"
            )
            {
                sessionOptions.AppendExecutionProvider_CoreML();
            }
            var model = new InferenceSession(modelPath, sessionOptions);
            Debug.Log($"[MuseTalkInference] Loaded {modelName} from {modelPath}");
            return model;
        }

        /// <summary>
        /// Get model file path with optimal quality/performance balance
        /// QUALITY OPTIMIZATION: Automatically use FP32 for VAE models to preserve image quality
        /// </summary>
        public static string GetModelPath(MuseTalkConfig config, string baseName)
        {
            // SPECIAL CASE: Whisper and Face Parsing models don't use version suffix
            bool isVersionIndependent = baseName.Contains("whisper_encoder") || 
                                        baseName.Contains("face_parsing") ||
                                        baseName.Contains("det_10g") ||
                                        baseName.Contains("1k3d68");
            
            string baseModelPath;
            if (isVersionIndependent)
            {
                baseModelPath = Path.Combine(config.ModelPath, $"{baseName}.onnx");
            }
            else
            {
                baseModelPath = Path.Combine(config.ModelPath, $"{baseName}_{config.Version}.onnx");
            }
            
            bool forceFP32 = false; // baseName.Contains("vae_decoder"); // VAE decoder quality is sensitive to FP32
            
            // For other models: Priority INT8 for CPU optimization > FP32 fallback
            if (config.UseINT8 && !forceFP32)
            {
                // Try INT8 model first (optimal for CPU performance)
                string int8ModelPath;
                if (isVersionIndependent)
                {
                    int8ModelPath = Path.Combine(config.ModelPath, $"{baseName}_int8.onnx");
                }
                else
                {
                    int8ModelPath = Path.Combine(config.ModelPath, $"{baseName}_{config.Version}_int8.onnx");
                }
                
                if (File.Exists(int8ModelPath))
                {
                    Debug.Log($"[MuseTalkInference] Using INT8 model (performance optimization): {int8ModelPath}");
                    return int8ModelPath;
                }
                Debug.LogWarning($"[MuseTalkInference] INT8 model not found: {int8ModelPath}, falling back to FP32");
            }
            
            Debug.Log($"[MuseTalkInference] Using FP32 model: {baseModelPath}");
            return baseModelPath;
        }
    }
} 