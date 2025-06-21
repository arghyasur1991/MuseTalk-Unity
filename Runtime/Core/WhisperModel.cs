using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace LiveTalk.Core
{
    using API;
    using Utils;
    
    /// <summary>
    /// Unity ONNX Whisper model for audio feature extraction matching Python implementation
    /// </summary>
    internal class WhisperModel : IDisposable
    {
        private InferenceSession _session;
        private bool _isInitialized = false;
        
        // Model constants matching Python implementation
        private const int SAMPLE_RATE = 16000;
        private const int N_MELS = 80;
        private const int HOP_LENGTH = 160;  // 10ms hop
        private const int WIN_LENGTH = 400;  // 25ms window
        private const int TARGET_FRAMES = 3000; // 30 seconds at 10ms hop
        
        // ONNX model input/output names
        private const string INPUT_NAME = "input_features";
        private const string OUTPUT_NAME = "audio_features_all_layers";
        
        public bool IsInitialized => _isInitialized;
        
        #if UNITY_EDITOR
        /// <summary>
        /// TESTING ONLY: Expose mel spectrogram extraction for comparison
        /// </summary>
        public float[,] TestExtractMelSpectrogram(float[] audioSamples, int sampleRate)
        {
            if (sampleRate != SAMPLE_RATE)
            {
                audioSamples = AudioUtils.ResampleAudio(audioSamples, sampleRate, SAMPLE_RATE);
            }
            return ExtractMelSpectrogram(audioSamples);
        }
        
        /// <summary>
        /// TESTING ONLY: Expose ONNX inference for comparison
        /// </summary>
        public float[,,,] TestRunWhisperInference(float[,] melSpectrogram)
        {
            return RunWhisperInference(melSpectrogram);
        }
        
        /// <summary>
        /// TESTING ONLY: Expose chunk processing for comparison
        /// </summary>
        public float[][] TestProcessWhisperChunks(float[,,,] whisperFeatures, int audioLength)
        {
            var audioFeatures = ProcessWhisperFeatures(whisperFeatures, audioLength);
            return audioFeatures.FeatureChunks.ToArray();
        }
        #endif
        
        /// <summary>
        /// Initialize Whisper model from StreamingAssets
        /// </summary>
        public WhisperModel(LiveTalkConfig config)
        {
            _session = ModelUtils.LoadModel(config, "whisper_encoder");
            _isInitialized = true;
            VerifyModelSignature();
        }
        
        private void VerifyModelSignature()
        {
            if (!_isInitialized || _session == null) return;
            
            var inputMeta = _session.InputMetadata;
            var outputMeta = _session.OutputMetadata;
            
            // Debug.Log($"[WhisperModel] Inputs: {string.Join(", ", inputMeta.Keys)}");
            // Debug.Log($"[WhisperModel] Outputs: {string.Join(", ", outputMeta.Keys)}");
            
            if (!inputMeta.ContainsKey(INPUT_NAME))
            {
                Debug.LogError($"[WhisperModel] Expected input '{INPUT_NAME}' not found");
            }
            
            if (!outputMeta.ContainsKey(OUTPUT_NAME))
            {
                Debug.LogError($"[WhisperModel] Expected output '{OUTPUT_NAME}' not found");
            }
        }
        
        /// <summary>
        /// Process audio samples and extract Whisper features using pure Unity/C# implementation
        /// Matches the Python onnx_inference.py processing exactly
        /// </summary>
        public AudioFeatures ProcessAudio(float[] audioSamples, int originalSampleRate = 44100)
        {
            if (!_isInitialized)
            {
                Debug.LogError("[WhisperModel] Model not initialized");
                return null;
            }
            
            if (audioSamples == null || audioSamples.Length == 0)
            {
                Debug.LogError("[WhisperModel] Audio samples are null or empty");
                return null;
            }
            
            try
            {
                // Step 1: Resample to 16kHz if needed
                float[] resampledAudio = audioSamples;
                if (originalSampleRate != SAMPLE_RATE)
                {
                    resampledAudio = AudioUtils.ResampleAudio(audioSamples, originalSampleRate, SAMPLE_RATE);
                }
                
                // Step 2: Extract mel spectrogram
                float[,] melSpectrogram = ExtractMelSpectrogram(resampledAudio);
                
                // Step 3: Process through ONNX Whisper
                var whisperFeatures = RunWhisperInference(melSpectrogram);
                
                // Step 4: Convert to MuseTalk audio chunks
                var audioFeatures = ProcessWhisperFeatures(whisperFeatures, resampledAudio.Length);
                
                return audioFeatures;
            }
            catch (Exception e)
            {
                Debug.LogError($"[WhisperModel] Error processing audio: {e.Message}");
                return null;
            }
        }
        
        /// <summary>
        /// Extract mel spectrogram matching Python librosa.feature.melspectrogram exactly
        /// </summary>
        private float[,] ExtractMelSpectrogram(float[] audioSamples)
        {
            // Use librosa-compatible parameters
            const int nFft = 512;  // Default librosa n_fft for 16kHz
            
            // Add padding for centering (matching librosa center=True)
            int halfWindow = nFft / 2;
            float[] paddedAudio = new float[audioSamples.Length + nFft - 1];
            
            // Reflect padding at the beginning - FIXED with bounds checking
            for (int i = 0; i < halfWindow; i++)
            {
                int srcIndex = halfWindow - 1 - i;
                // Bounds checking: ensure srcIndex is valid
                if (srcIndex >= 0 && srcIndex < audioSamples.Length)
                {
                    paddedAudio[i] = audioSamples[srcIndex];
                }
                else
                {
                    // If audio is too short, repeat the first sample
                    paddedAudio[i] = audioSamples.Length > 0 ? audioSamples[0] : 0f;
                }
            }
            
            // Copy original audio
            Array.Copy(audioSamples, 0, paddedAudio, halfWindow, audioSamples.Length);
            
            // Reflect padding at the end - FIXED with bounds checking
            for (int i = 0; i < halfWindow; i++)
            {
                int srcIndex = audioSamples.Length - 1 - i;
                int destIndex = halfWindow + audioSamples.Length + i;
                
                // Bounds checking: ensure indices are valid
                if (srcIndex >= 0 && srcIndex < audioSamples.Length && destIndex < paddedAudio.Length)
                {
                    paddedAudio[destIndex] = audioSamples[srcIndex];
                }
                else
                {
                    // If audio is too short, repeat the last sample
                    float lastSample = audioSamples.Length > 0 ? audioSamples[audioSamples.Length - 1] : 0f;
                    if (destIndex < paddedAudio.Length)
                    {
                        paddedAudio[destIndex] = lastSample;
                    }
                }
            }
            
            // Calculate number of frames (matching librosa with centering)
            int numFrames = (paddedAudio.Length - nFft) / HOP_LENGTH + 1;
            numFrames = Mathf.Min(numFrames, TARGET_FRAMES);
            
            // Create mel filterbank (matching librosa)
            float[,] melFilterBank = CreateLibrosaMelFilterBank(nFft);
            
            // Compute power spectrogram
            float[,] powerSpec = new float[nFft / 2 + 1, numFrames];
            
            for (int frame = 0; frame < numFrames; frame++)
            {
                int startSample = frame * HOP_LENGTH;
                
                // Apply Hann window (matching librosa)
                float[] windowedFrame = new float[nFft];
                for (int i = 0; i < nFft && startSample + i < paddedAudio.Length; i++)
                {
                    float hannWindow = 0.5f * (1f - Mathf.Cos(2f * Mathf.PI * i / (nFft - 1)));
                    windowedFrame[i] = paddedAudio[startSample + i] * hannWindow;
                }
                
                // Compute FFT magnitude squared (power)
                for (int bin = 0; bin < nFft / 2 + 1; bin++)
                {
                    float real = 0f, imag = 0f;
                    
                    for (int i = 0; i < nFft; i++)
                    {
                        float angle = -2f * Mathf.PI * bin * i / nFft;
                        real += windowedFrame[i] * Mathf.Cos(angle);
                        imag += windowedFrame[i] * Mathf.Sin(angle);
                    }
                    
                    powerSpec[bin, frame] = real * real + imag * imag;
                }
            }
            
            // Apply mel filterbank to get mel spectrogram
            float[,] melSpec = new float[N_MELS, numFrames];
            
            for (int mel = 0; mel < N_MELS; mel++)
            {
                for (int frame = 0; frame < numFrames; frame++)
                {
                    float melValue = 0f;
                    
                    for (int bin = 0; bin < nFft / 2 + 1; bin++)
                    {
                        melValue += melFilterBank[mel, bin] * powerSpec[bin, frame];
                    }
                    
                    melSpec[mel, frame] = melValue;
                }
            }
            
            // Convert to log scale (matching librosa.power_to_db)
            float maxValue = float.MinValue;
            for (int mel = 0; mel < N_MELS; mel++)
            {
                for (int frame = 0; frame < numFrames; frame++)
                {
                    maxValue = Mathf.Max(maxValue, melSpec[mel, frame]);
                }
            }
            
            for (int mel = 0; mel < N_MELS; mel++)
            {
                for (int frame = 0; frame < numFrames; frame++)
                {
                    // power_to_db: 10 * log10(max(S, ref)) where ref = max(S)
                    float dbValue = 10f * Mathf.Log10(Mathf.Max(melSpec[mel, frame], maxValue * 1e-10f)) 
                                  - 10f * Mathf.Log10(maxValue);
                    
                    // Apply top_db=80.0 clamping (matching librosa exactly)
                    melSpec[mel, frame] = Mathf.Max(dbValue, -80f);
                }
            }
            
            // Normalize exactly like Python: (mel + 80.0) / 80.0, clipped to [-1, 1]
            for (int mel = 0; mel < N_MELS; mel++)
            {
                for (int frame = 0; frame < numFrames; frame++)
                {
                    melSpec[mel, frame] = (melSpec[mel, frame] + 80f) / 80f;
                    melSpec[mel, frame] = Mathf.Clamp(melSpec[mel, frame], -1f, 1f);
                }
            }
            
            // Pad to target frames if needed
            if (numFrames < TARGET_FRAMES)
            {
                float[,] paddedMel = new float[N_MELS, TARGET_FRAMES];
                for (int mel = 0; mel < N_MELS; mel++)
                {
                    for (int frame = 0; frame < TARGET_FRAMES; frame++)
                    {
                        paddedMel[mel, frame] = frame < numFrames ? melSpec[mel, frame] : 0f;
                    }
                }
                return paddedMel;
            }
            
            return melSpec;
        }
        
        /// <summary>
        /// Create librosa-compatible mel filterbank (FIXED to match librosa exactly)
        /// </summary>
        private float[,] CreateLibrosaMelFilterBank(int nFft)
        {
            int nFreqBins = nFft / 2 + 1;
            float[,] melFilters = new float[N_MELS, nFreqBins];
            
            // Create mel frequency points (matching librosa exactly)
            float melMin = HzToMel(0f);
            float melMax = HzToMel(8000f); // Nyquist for 16kHz
            
            float[] melPoints = new float[N_MELS + 2];
            for (int i = 0; i < melPoints.Length; i++)
            {
                float mel = melMin + (melMax - melMin) * i / (melPoints.Length - 1);
                melPoints[i] = MelToHz(mel);
            }
            
            // Convert mel frequencies to FFT bin indices
            float[] binPoints = new float[N_MELS + 2];
            for (int i = 0; i < binPoints.Length; i++)
            {
                binPoints[i] = melPoints[i] * nFft / SAMPLE_RATE;
            }
            
            // Create triangular filters (FIXED: proper normalization like librosa)
            for (int mel = 0; mel < N_MELS; mel++)
            {
                float leftBin = binPoints[mel];
                float centerBin = binPoints[mel + 1];
                float rightBin = binPoints[mel + 2];
                
                // Calculate normalization factor (matching librosa)
                float leftWidth = centerBin - leftBin;
                float rightWidth = rightBin - centerBin;
                
                for (int bin = 0; bin < nFreqBins; bin++)
                {
                    if (bin >= leftBin && bin <= centerBin && leftWidth > 0)
                    {
                        // FIXED: Normalize by 2.0 / (rightBin - leftBin) like librosa
                        melFilters[mel, bin] = 2.0f * (bin - leftBin) / ((rightBin - leftBin) * leftWidth);
                    }
                    else if (bin > centerBin && bin <= rightBin && rightWidth > 0)
                    {
                        // FIXED: Normalize by 2.0 / (rightBin - leftBin) like librosa  
                        melFilters[mel, bin] = 2.0f * (rightBin - bin) / ((rightBin - leftBin) * rightWidth);
                    }
                    else
                    {
                        melFilters[mel, bin] = 0f;
                    }
                }
            }
            
            return melFilters;
        }
        
        /// <summary>
        /// Convert Hz to mel scale
        /// </summary>
        private float HzToMel(float hz)
        {
            return 2595f * Mathf.Log10(1f + hz / 700f);
        }
        
        /// <summary>
        /// Convert mel scale to Hz
        /// </summary>
        private float MelToHz(float mel)
        {
            return 700f * (Mathf.Pow(10f, mel / 2595f) - 1f);
        }
        
        /// <summary>
        /// Run the ONNX Whisper model inference
        /// </summary>
        private float[,,,] RunWhisperInference(float[,] melSpectrogram)
        {
            int melBands = melSpectrogram.GetLength(0);
            int frames = melSpectrogram.GetLength(1);
            
            // Pad to target frames
            float[,,] inputTensor = new float[1, melBands, TARGET_FRAMES];
            
            for (int mel = 0; mel < melBands; mel++)
            {
                for (int frame = 0; frame < TARGET_FRAMES; frame++)
                {
                    if (frame < frames)
                    {
                        inputTensor[0, mel, frame] = melSpectrogram[mel, frame];
                    }
                    // else: padding with zeros (default initialization)
                }
            }
            
            // Create ONNX tensor
            var inputShape = new int[] { 1, melBands, TARGET_FRAMES };
            var tensor = new DenseTensor<float>(inputTensor.Cast<float>().ToArray(), inputShape);
            
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(INPUT_NAME, tensor)
            };
            
            using var outputs = _session.Run(inputs);
            var output = outputs.First(o => o.Name == OUTPUT_NAME);
            
            if (output.Value is DenseTensor<float> outputTensor)
            {
                // Expected shape: [batch, seq_len, layers, features] = [1, seq_len, layers, 384]
                var shape = outputTensor.Dimensions.ToArray();

                
                // Convert to 4D array for easier processing
                int batchSize = (int)shape[0];
                int seqLen = (int)shape[1];
                int layers = (int)shape[2];
                int features = (int)shape[3];
                
                float[,,,] result = new float[batchSize, seqLen, layers, features];
                var buffer = outputTensor.Buffer.ToArray();
                
                for (int b = 0; b < batchSize; b++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        for (int l = 0; l < layers; l++)
                        {
                            for (int f = 0; f < features; f++)
                            {
                                int index = b * seqLen * layers * features + 
                                           s * layers * features + 
                                           l * features + f;
                                result[b, s, l, f] = buffer[index];
                            }
                        }
                    }
                }
                
                return result;
            }
            
            throw new InvalidOperationException("Failed to get valid output tensor from Whisper ONNX");
        }
        
        /// <summary>
        /// Process Whisper features into MuseTalk audio chunks
        /// </summary>
        private AudioFeatures ProcessWhisperFeatures(float[,,,] whisperFeatures, int audioLength)
        {
            // Constants from Python implementation
            const int fps = 25;
            const int audioFps = 50;
            const int audioPaddingLeft = 2;
            const int audioPaddingRight = 2;
            
            // Get dimensions [batch, seq_len, layers, features]
            int seqLen = whisperFeatures.GetLength(1);
            int layers = whisperFeatures.GetLength(2);
            int features = whisperFeatures.GetLength(3);
            
            // Calculate parameters
            float whisperIdxMultiplier = (float)audioFps / fps;
            int numFrames = Mathf.FloorToInt((float)audioLength / SAMPLE_RATE * fps);
            int actualLength = Mathf.FloorToInt((float)audioLength / SAMPLE_RATE * audioFps);
            
            // Trim to actual length
            actualLength = Mathf.Min(actualLength, seqLen);
            
            // Add padding
            int paddingNums = Mathf.CeilToInt(whisperIdxMultiplier);
            int leftPaddingSize = paddingNums * audioPaddingLeft;
            int rightPaddingSize = paddingNums * 3 * audioPaddingRight;
            
            int totalPaddedLength = leftPaddingSize + actualLength + rightPaddingSize;
            

            
            // Create padded features array
            float[,,,] paddedFeatures = new float[1, totalPaddedLength, layers, features];
            
            // Copy actual features to padded array (padding is zeros by default)
            for (int s = 0; s < actualLength; s++)
            {
                for (int l = 0; l < layers; l++)
                {
                    for (int f = 0; f < features; f++)
                    {
                        paddedFeatures[0, leftPaddingSize + s, l, f] = whisperFeatures[0, s, l, f];
                    }
                }
            }
            
            // Generate chunks
            int audioFeatureLengthPerFrame = 2 * (audioPaddingLeft + audioPaddingRight + 1);
            var featureChunks = new List<float[]>();
            

            
            for (int frameIndex = 0; frameIndex < numFrames; frameIndex++)
            {
                int audioIndex = Mathf.FloorToInt(frameIndex * whisperIdxMultiplier);
                
                if (audioIndex + audioFeatureLengthPerFrame <= totalPaddedLength)
                {
                    // Extract chunk and reshape to match Python
                    int chunkHeight = audioFeatureLengthPerFrame * layers;
                    float[] chunk = new float[chunkHeight * features];
                    
                    for (int t = 0; t < audioFeatureLengthPerFrame; t++)
                    {
                        for (int l = 0; l < layers; l++)
                        {
                            for (int f = 0; f < features; f++)
                            {
                                int rowIndex = t * layers + l;
                                int chunkIndex = rowIndex * features + f;
                                chunk[chunkIndex] = paddedFeatures[0, audioIndex + t, l, f];
                            }
                        }
                    }
                    
                    featureChunks.Add(chunk);
                }
            }
            
            var audioFeatures = new AudioFeatures
            {
                FeatureChunks = featureChunks,
                SampleRate = SAMPLE_RATE,
                Duration = (float)audioLength / SAMPLE_RATE
            };
            
            int expectedChunkSize = audioFeatureLengthPerFrame * layers * features;
            
            return audioFeatures;
        }
        
        public void Dispose()
        {
            _session?.Dispose();
            _session = null;
            _isInitialized = false;
            Debug.Log("[WhisperModel] Disposed");
        }
    }
} 