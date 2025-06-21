using System;
using UnityEngine;

namespace LiveTalk.Utils
{
    /// <summary>
    /// Utility functions for audio processing in MuseTalk
    /// </summary>
    public static class AudioUtils
    {
        /// <summary>
        /// Convert AudioClip to float array
        /// </summary>
        public static float[] AudioClipToFloatArray(AudioClip audioClip)
        {
            if (audioClip == null)
                throw new ArgumentNullException(nameof(audioClip));
                
            float[] samples = new float[audioClip.samples * audioClip.channels];
            audioClip.GetData(samples, 0);
            
            return samples;
        }
        
        /// <summary>
        /// Create AudioClip from float array
        /// </summary>
        public static AudioClip FloatArrayToAudioClip(float[] samples, int sampleRate, int channels = 1, string name = "GeneratedAudio")
        {
            if (samples == null)
                throw new ArgumentNullException(nameof(samples));
                
            int sampleCount = samples.Length / channels;
            AudioClip clip = AudioClip.Create(name, sampleCount, channels, sampleRate, false);
            clip.SetData(samples, 0);
            
            return clip;
        }
        
        /// <summary>
        /// Resample audio to target sample rate
        /// </summary>
        public static float[] ResampleAudio(float[] inputSamples, int originalSampleRate, int targetSampleRate)
        {
            if (inputSamples == null)
                throw new ArgumentNullException(nameof(inputSamples));
                
            if (originalSampleRate == targetSampleRate)
                return inputSamples;
                
            float ratio = (float)targetSampleRate / originalSampleRate;
            int outputLength = Mathf.RoundToInt(inputSamples.Length * ratio);
            float[] outputSamples = new float[outputLength];
            
            for (int i = 0; i < outputLength; i++)
            {
                float sourceIndex = i / ratio;
                int index = Mathf.FloorToInt(sourceIndex);
                float fraction = sourceIndex - index;
                
                if (index < inputSamples.Length - 1)
                {
                    // Linear interpolation
                    outputSamples[i] = Mathf.Lerp(inputSamples[index], inputSamples[index + 1], fraction);
                }
                else if (index < inputSamples.Length)
                {
                    outputSamples[i] = inputSamples[index];
                }
            }
            
            return outputSamples;
        }
        
        /// <summary>
        /// Convert stereo audio to mono by averaging channels
        /// </summary>
        public static float[] StereoToMono(float[] stereoSamples)
        {
            if (stereoSamples == null)
                throw new ArgumentNullException(nameof(stereoSamples));
                
            if (stereoSamples.Length % 2 != 0)
                throw new ArgumentException("Stereo samples array length must be even");
                
            float[] monoSamples = new float[stereoSamples.Length / 2];
            
            for (int i = 0; i < monoSamples.Length; i++)
            {
                monoSamples[i] = (stereoSamples[i * 2] + stereoSamples[i * 2 + 1]) * 0.5f;
            }
            
            return monoSamples;
        }
        
        /// <summary>
        /// Normalize audio samples to [-1, 1] range
        /// </summary>
        public static float[] NormalizeAudio(float[] samples)
        {
            if (samples == null)
                throw new ArgumentNullException(nameof(samples));
                
            float maxValue = 0f;
            
            // Find maximum absolute value
            for (int i = 0; i < samples.Length; i++)
            {
                float absValue = Mathf.Abs(samples[i]);
                if (absValue > maxValue)
                    maxValue = absValue;
            }
            
            // Normalize if needed
            if (maxValue > 0f && maxValue != 1f)
            {
                float[] normalizedSamples = new float[samples.Length];
                float normalizationFactor = 1f / maxValue;
                
                for (int i = 0; i < samples.Length; i++)
                {
                    normalizedSamples[i] = samples[i] * normalizationFactor;
                }
                
                return normalizedSamples;
            }
            
            return samples;
        }
        
        /// <summary>
        /// Apply a simple highpass filter to remove low-frequency noise
        /// </summary>
        public static float[] ApplyHighpassFilter(float[] samples, float cutoffFrequency, int sampleRate)
        {
            if (samples == null)
                throw new ArgumentNullException(nameof(samples));
                
            // Simple first-order highpass filter
            float rc = 1.0f / (cutoffFrequency * 2.0f * Mathf.PI);
            float dt = 1.0f / sampleRate;
            float alpha = rc / (rc + dt);
            
            float[] filteredSamples = new float[samples.Length];
            filteredSamples[0] = samples[0];
            
            for (int i = 1; i < samples.Length; i++)
            {
                filteredSamples[i] = alpha * (filteredSamples[i - 1] + samples[i] - samples[i - 1]);
            }
            
            return filteredSamples;
        }
        
        /// <summary>
        /// Calculate RMS (Root Mean Square) of audio samples
        /// </summary>
        public static float CalculateRMS(float[] samples)
        {
            if (samples == null || samples.Length == 0)
                return 0f;
                
            float sum = 0f;
            for (int i = 0; i < samples.Length; i++)
            {
                sum += samples[i] * samples[i];
            }
            
            return Mathf.Sqrt(sum / samples.Length);
        }
        
        /// <summary>
        /// Detect if audio contains speech (simple energy-based detection)
        /// </summary>
        public static bool ContainsSpeech(float[] samples, float threshold = 0.01f)
        {
            float rms = CalculateRMS(samples);
            return rms > threshold;
        }
        
        /// <summary>
        /// Trim silence from the beginning and end of audio
        /// </summary>
        public static float[] TrimSilence(float[] samples, float threshold = 0.001f)
        {
            if (samples == null || samples.Length == 0)
                return samples;
                
            int startIndex = 0;
            int endIndex = samples.Length - 1;
            
            // Find start of audio content
            for (int i = 0; i < samples.Length; i++)
            {
                if (Mathf.Abs(samples[i]) > threshold)
                {
                    startIndex = i;
                    break;
                }
            }
            
            // Find end of audio content
            for (int i = samples.Length - 1; i >= 0; i--)
            {
                if (Mathf.Abs(samples[i]) > threshold)
                {
                    endIndex = i;
                    break;
                }
            }
            
            if (startIndex >= endIndex)
                return new float[0]; // All silence
                
            int trimmedLength = endIndex - startIndex + 1;
            float[] trimmedSamples = new float[trimmedLength];
            Array.Copy(samples, startIndex, trimmedSamples, 0, trimmedLength);
            
            return trimmedSamples;
        }
    }
} 