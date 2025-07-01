package com.taobao.android.utils;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class WavFileGenerator {
    
    private static final String TAG = "WavFileGenerator";
    
    /**
     * 生成示例WAV文件
     * @param context 上下文
     * @param fileName 文件名
     * @return 成功返回true
     */
    public static boolean generateExampleWav(Context context, String fileName) {
        try {
            File file = new File(context.getCacheDir(), fileName);
            
            // 音频参数
            int sampleRate = 16000;   // 16kHz采样率
            int duration = 5;         // 5秒时长
            int channels = 1;         // 单声道
            int bitsPerSample = 16;   // 16位
            
            int totalSamples = sampleRate * duration;
            int bytesPerSample = bitsPerSample / 8;
            int dataSize = totalSamples * channels * bytesPerSample;
            int fileSize = 36 + dataSize;
            
            FileOutputStream fos = new FileOutputStream(file);
            
            // 写入WAV头部
            writeWavHeader(fos, fileSize, sampleRate, channels, bitsPerSample, dataSize);
            
            // 生成音频数据（简单的正弦波混合）
            generateAudioData(fos, totalSamples, sampleRate);
            
            fos.close();
            
            Log.d(TAG, "生成示例WAV文件成功: " + file.getAbsolutePath());
            Log.d(TAG, "文件大小: " + file.length() + " bytes");
            
            return true;
            
        } catch (IOException e) {
            Log.e(TAG, "生成WAV文件失败", e);
            return false;
        }
    }
    
    /**
     * 写入WAV文件头部
     */
    private static void writeWavHeader(FileOutputStream fos, int fileSize, int sampleRate, 
                                     int channels, int bitsPerSample, int dataSize) throws IOException {
        
        ByteBuffer buffer = ByteBuffer.allocate(44);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        // RIFF chunk
        buffer.put("RIFF".getBytes());
        buffer.putInt(fileSize - 8);
        buffer.put("WAVE".getBytes());
        
        // fmt chunk
        buffer.put("fmt ".getBytes());
        buffer.putInt(16);  // fmt chunk size
        buffer.putShort((short) 1);  // PCM format
        buffer.putShort((short) channels);
        buffer.putInt(sampleRate);
        buffer.putInt(sampleRate * channels * bitsPerSample / 8);  // byte rate
        buffer.putShort((short) (channels * bitsPerSample / 8));  // block align
        buffer.putShort((short) bitsPerSample);
        
        // data chunk
        buffer.put("data".getBytes());
        buffer.putInt(dataSize);
        
        fos.write(buffer.array());
    }
    
    /**
     * 生成音频数据（模拟语音信号）
     */
    private static void generateAudioData(FileOutputStream fos, int totalSamples, int sampleRate) throws IOException {
        
        ByteBuffer buffer = ByteBuffer.allocate(totalSamples * 2);  // 16位 = 2字节
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        // 生成混合频率的信号，模拟语音
        for (int i = 0; i < totalSamples; i++) {
            double time = (double) i / sampleRate;
            
            // 基础频率成分（模拟语音基频）
            double signal = 0.0;
            signal += 0.3 * Math.sin(2 * Math.PI * 150 * time);    // 150Hz基频
            signal += 0.2 * Math.sin(2 * Math.PI * 300 * time);    // 300Hz倍频
            signal += 0.15 * Math.sin(2 * Math.PI * 450 * time);   // 450Hz倍频
            signal += 0.1 * Math.sin(2 * Math.PI * 600 * time);    // 600Hz倍频
            
            // 添加一些高频成分
            signal += 0.05 * Math.sin(2 * Math.PI * 1200 * time);
            signal += 0.03 * Math.sin(2 * Math.PI * 2400 * time);
            
            // 添加调制（模拟语音变化）
            double modulation = 1.0 + 0.3 * Math.sin(2 * Math.PI * 5 * time);  // 5Hz调制
            signal *= modulation;
            
            // 添加包络（避免突然开始和结束）
            double envelope = 1.0;
            if (time < 0.1) {
                envelope = time / 0.1;  // 淡入
            } else if (time > 4.9) {
                envelope = (5.0 - time) / 0.1;  // 淡出
            }
            signal *= envelope;
            
            // 添加少量噪声
            signal += 0.02 * (Math.random() - 0.5);
            
            // 转换为16位整数
            signal = Math.max(-1.0, Math.min(1.0, signal));  // 限幅
            short sample = (short) (signal * 32767);
            
            buffer.putShort(sample);
        }
        
        fos.write(buffer.array());
    }
} 