package com.taobao.android.utils;

import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class AudioPlayer {
    
    private static final String TAG = "AudioPlayer";
    private static final int SAMPLE_RATE = 16000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_OUT_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    
    private AudioTrack mAudioTrack;
    private boolean mIsPlaying = false;
    
    /**
     * 播放浮点音频数据
     * @param audioData 浮点音频数据 [-1.0, 1.0]
     */
    public void playAudioData(float[] audioData) {
        if (audioData == null || audioData.length == 0) {
            Log.e(TAG, "音频数据为空");
            return;
        }
        
        // 转换为16位PCM
        short[] pcmData = floatToPcm16(audioData);
        
        // 创建AudioTrack
        int bufferSize = AudioTrack.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        bufferSize = Math.max(bufferSize, pcmData.length * 2); // 确保缓冲区足够大
        
        mAudioTrack = new AudioTrack(
            AudioManager.STREAM_MUSIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            bufferSize,
            AudioTrack.MODE_STREAM
        );
        
        if (mAudioTrack.getState() != AudioTrack.STATE_INITIALIZED) {
            Log.e(TAG, "AudioTrack初始化失败");
            return;
        }
        
        // 播放音频
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    mIsPlaying = true;
                    mAudioTrack.play();
                    
                    // 写入音频数据
                    byte[] buffer = new byte[pcmData.length * 2];
                    ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(pcmData);
                    
                    int bytesWritten = mAudioTrack.write(buffer, 0, buffer.length);
                    Log.d(TAG, "音频播放完成，写入字节数: " + bytesWritten);
                    
                    // 等待播放完成
                    while (mAudioTrack.getPlaybackHeadPosition() < pcmData.length && mIsPlaying) {
                        Thread.sleep(10);
                    }
                    
                } catch (Exception e) {
                    Log.e(TAG, "音频播放异常", e);
                } finally {
                    stopPlayback();
                }
            }
        }).start();
    }
    
    /**
     * 停止播放
     */
    public void stopPlayback() {
        mIsPlaying = false;
        if (mAudioTrack != null) {
            try {
                if (mAudioTrack.getState() == AudioTrack.STATE_INITIALIZED) {
                    mAudioTrack.stop();
                }
                mAudioTrack.release();
            } catch (Exception e) {
                Log.e(TAG, "停止播放异常", e);
            }
            mAudioTrack = null;
        }
    }
    
    /**
     * 保存音频数据为WAV文件
     * @param audioData 浮点音频数据
     * @param outputPath 输出文件路径
     * @return 是否保存成功
     */
    public static boolean saveAsWav(float[] audioData, String outputPath) {
        if (audioData == null || audioData.length == 0) {
            Log.e(TAG, "音频数据为空，无法保存");
            return false;
        }
        
        try {
            // 转换为16位PCM
            short[] pcmData = floatToPcm16(audioData);
            
            // 写入WAV文件
            FileOutputStream fos = new FileOutputStream(outputPath);
            
            // WAV文件头
            writeWavHeader(fos, pcmData.length, SAMPLE_RATE, 1, 16);
            
            // 写入PCM数据
            byte[] buffer = new byte[pcmData.length * 2];
            ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(pcmData);
            fos.write(buffer);
            
            fos.close();
            
            Log.d(TAG, "音频保存成功: " + outputPath + ", 时长: " + (audioData.length / (float)SAMPLE_RATE) + "秒");
            return true;
            
        } catch (IOException e) {
            Log.e(TAG, "保存音频文件失败: " + outputPath, e);
            return false;
        }
    }
    
    /**
     * 浮点音频转换为16位PCM
     * @param audioData 浮点音频数据 [-1.0, 1.0]
     * @return 16位PCM数据
     */
    private static short[] floatToPcm16(float[] audioData) {
        short[] pcmData = new short[audioData.length];
        for (int i = 0; i < audioData.length; i++) {
            // 限制范围并转换
            float sample = Math.max(-1.0f, Math.min(1.0f, audioData[i]));
            pcmData[i] = (short) (sample * 32767);
        }
        return pcmData;
    }
    
    /**
     * 写入WAV文件头
     */
    private static void writeWavHeader(FileOutputStream fos, int dataLength, int sampleRate, 
                                      int channels, int bitsPerSample) throws IOException {
        int byteRate = sampleRate * channels * bitsPerSample / 8;
        int blockAlign = channels * bitsPerSample / 8;
        int dataSize = dataLength * blockAlign;
        int fileSize = 36 + dataSize;
        
        ByteBuffer buffer = ByteBuffer.allocate(44);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        
        // RIFF头
        buffer.put("RIFF".getBytes());
        buffer.putInt(fileSize);
        buffer.put("WAVE".getBytes());
        
        // fmt chunk
        buffer.put("fmt ".getBytes());
        buffer.putInt(16); // fmt chunk size
        buffer.putShort((short) 1); // PCM format
        buffer.putShort((short) channels);
        buffer.putInt(sampleRate);
        buffer.putInt(byteRate);
        buffer.putShort((short) blockAlign);
        buffer.putShort((short) bitsPerSample);
        
        // data chunk
        buffer.put("data".getBytes());
        buffer.putInt(dataSize);
        
        fos.write(buffer.array());
    }
    
    /**
     * 检查是否正在播放
     */
    public boolean isPlaying() {
        return mIsPlaying;
    }
} 