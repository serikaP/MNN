package com.taobao.android.utils;

import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class AudioFileReader {
    
    private static final String TAG = "AudioFileReader";
    
    /**
     * 加载WAV文件并转换为float数组
     * @param filePath WAV文件路径
     * @param targetSampleRate 目标采样率（默认16000）
     * @return 音频数据的float数组，失败返回null
     */
    public static float[] loadWavFile(String filePath, int targetSampleRate) {
        try {
            FileInputStream fis = new FileInputStream(filePath);
            float[] audioData = loadWavFromStream(fis, targetSampleRate);
            fis.close();
            return audioData;
        } catch (IOException e) {
            Log.e(TAG, "加载WAV文件失败: " + filePath, e);
            return null;
        }
    }
    
    /**
     * 从InputStream加载WAV数据
     * @param inputStream 输入流
     * @param targetSampleRate 目标采样率
     * @return 音频数据的float数组
     * @throws IOException 读取异常
     */
    public static float[] loadWavFromStream(InputStream inputStream, int targetSampleRate) throws IOException {
        // 读取整个文件到字节数组
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int bytesRead;
        while ((bytesRead = inputStream.read(buffer)) != -1) {
            baos.write(buffer, 0, bytesRead);
        }
        byte[] wavData = baos.toByteArray();
        baos.close();
        
        return parseWavData(wavData, targetSampleRate);
    }
    
    /**
     * 解析WAV文件数据
     * @param wavData WAV文件的字节数据
     * @param targetSampleRate 目标采样率
     * @return 音频数据的float数组
     */
    private static float[] parseWavData(byte[] wavData, int targetSampleRate) {
        try {
            ByteBuffer buffer = ByteBuffer.wrap(wavData);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            
            // 读取WAV头部信息
            WavHeader header = parseWavHeader(buffer);
            if (header == null) {
                Log.e(TAG, "WAV头部解析失败");
                return null;
            }
            
            Log.d(TAG, "WAV文件信息:");
            Log.d(TAG, "  采样率: " + header.sampleRate + " Hz");
            Log.d(TAG, "  通道数: " + header.numChannels);
            Log.d(TAG, "  位深度: " + header.bitsPerSample + " bits");
            Log.d(TAG, "  数据长度: " + header.dataSize + " bytes");
            
            // 跳转到数据部分
            buffer.position(header.dataOffset);
            
            // 计算样本数量
            int bytesPerSample = header.bitsPerSample / 8;
            int totalSamples = header.dataSize / (bytesPerSample * header.numChannels);
            
            // 读取音频数据
            float[] audioSamples = new float[totalSamples];
            
            for (int i = 0; i < totalSamples; i++) {
                float sample = 0;
                
                // 处理多通道（转为单声道）
                for (int ch = 0; ch < header.numChannels; ch++) {
                    float channelSample = 0;
                    
                    if (header.bitsPerSample == 16) {
                        // 16位PCM
                        short value = buffer.getShort();
                        channelSample = value / 32768.0f;
                    } else if (header.bitsPerSample == 24) {
                        // 24位PCM
                        int value = (buffer.get() & 0xFF) | 
                                   ((buffer.get() & 0xFF) << 8) | 
                                   ((buffer.get() & 0xFF) << 16);
                        if (value >= 0x800000) {
                            value -= 0x1000000;
                        }
                        channelSample = value / 8388608.0f;
                    } else if (header.bitsPerSample == 32) {
                        // 32位PCM
                        int value = buffer.getInt();
                        channelSample = value / 2147483648.0f;
                    } else {
                        Log.e(TAG, "不支持的位深度: " + header.bitsPerSample);
                        return null;
                    }
                    
                    sample += channelSample;
                }
                
                // 多通道取平均
                if (header.numChannels > 1) {
                    sample /= header.numChannels;
                }
                
                audioSamples[i] = sample;
            }
            
            // 如果采样率不匹配，进行简单的重采样
            if (header.sampleRate != targetSampleRate) {
                audioSamples = resample(audioSamples, header.sampleRate, targetSampleRate);
                Log.d(TAG, "重采样: " + header.sampleRate + "Hz -> " + targetSampleRate + "Hz");
            }
            
            Log.d(TAG, "音频加载完成，样本数: " + audioSamples.length);
            Log.d(TAG, "音频时长: " + (audioSamples.length / (float)targetSampleRate) + " 秒");
            
            return audioSamples;
            
        } catch (Exception e) {
            Log.e(TAG, "WAV数据解析异常", e);
            return null;
        }
    }
    
    /**
     * 解析WAV文件头部
     * @param buffer 字节缓冲区
     * @return WAV头部信息
     */
    private static WavHeader parseWavHeader(ByteBuffer buffer) {
        try {
            WavHeader header = new WavHeader();
            
            // RIFF头部
            byte[] riffChunk = new byte[4];
            buffer.get(riffChunk);
            if (!"RIFF".equals(new String(riffChunk))) {
                Log.e(TAG, "不是有效的RIFF文件");
                return null;
            }
            
            int fileSize = buffer.getInt();
            
            byte[] waveChunk = new byte[4];
            buffer.get(waveChunk);
            if (!"WAVE".equals(new String(waveChunk))) {
                Log.e(TAG, "不是有效的WAVE文件");
                return null;
            }
            
            // 查找fmt chunk
            while (buffer.remaining() > 8) {
                byte[] chunkId = new byte[4];
                buffer.get(chunkId);
                int chunkSize = buffer.getInt();
                
                String chunkIdStr = new String(chunkId);
                
                if ("fmt ".equals(chunkIdStr)) {
                    // fmt chunk
                    short audioFormat = buffer.getShort();
                    header.numChannels = buffer.getShort();
                    header.sampleRate = buffer.getInt();
                    int byteRate = buffer.getInt();
                    short blockAlign = buffer.getShort();
                    header.bitsPerSample = buffer.getShort();
                    
                    // 跳过剩余的fmt数据
                    if (chunkSize > 16) {
                        buffer.position(buffer.position() + chunkSize - 16);
                    }
                    
                } else if ("data".equals(chunkIdStr)) {
                    // data chunk
                    header.dataSize = chunkSize;
                    header.dataOffset = buffer.position();
                    break;
                    
                } else {
                    // 跳过其他chunk
                    buffer.position(buffer.position() + chunkSize);
                }
            }
            
            if (header.dataSize == 0) {
                Log.e(TAG, "未找到数据chunk");
                return null;
            }
            
            return header;
            
        } catch (Exception e) {
            Log.e(TAG, "WAV头部解析异常", e);
            return null;
        }
    }
    
    /**
     * 简单的线性插值重采样
     * @param input 输入音频数据
     * @param inputRate 输入采样率
     * @param outputRate 输出采样率
     * @return 重采样后的音频数据
     */
    private static float[] resample(float[] input, int inputRate, int outputRate) {
        if (inputRate == outputRate) {
            return input;
        }
        
        double ratio = (double) inputRate / outputRate;
        int outputLength = (int) (input.length / ratio);
        float[] output = new float[outputLength];
        
        for (int i = 0; i < outputLength; i++) {
            double srcIndex = i * ratio;
            int index1 = (int) srcIndex;
            int index2 = Math.min(index1 + 1, input.length - 1);
            double fraction = srcIndex - index1;
            
            // 线性插值
            output[i] = (float) (input[index1] * (1 - fraction) + input[index2] * fraction);
        }
        
        return output;
    }
    
    /**
     * WAV文件头部信息类
     */
    private static class WavHeader {
        int numChannels;
        int sampleRate;
        int bitsPerSample;
        int dataSize;
        int dataOffset;
    }
} 