package com.taobao.android.mnndemo;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.SeekBar;

import com.taobao.android.mnn.MNNForwardType;
import com.taobao.android.mnn.MNNNetInstance;
import com.taobao.android.utils.AudioFileReader;
import com.taobao.android.utils.AudioPlayer;
import com.taobao.android.utils.Common;
import com.taobao.android.utils.PermissionUtils;
import com.taobao.android.utils.WavFileGenerator;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class AudioCodecActivity extends AppCompatActivity implements View.OnClickListener {

    private static final String TAG = "AudioCodecActivity";
    private static final String MODEL_FILE_NAME = "AudioCodec/funcodec_encoder.mnn";
    private static final String DECODER_MODEL_FILE_NAME = "AudioCodec/funcodec_decoder.mnn";
    private static final String EXAMPLE_WAV_NAME = "AudioCodec/example.wav";
    
    // 请求码
    private static final int REQUEST_IMPORT_WAV = 200;
    private static final int REQUEST_IMPORT_CODES = 201;
    
    private TextView mStatusText;
    private TextView mResultText;
    private TextView mTimeText;
    private Button mEncodeButton;
    private Button mImportWavButton;
    private Button mShareResultButton;
    private Button mImportCodesButton;
    private Button mDecodeButton;
    private Button mPlayAudioButton;
    private Button mSaveAudioButton;
    private SeekBar mQuantizerSeekBar;
    private TextView mQuantizerValueText;

    // 编码器相关
    private MNNNetInstance mNetInstance;
    private MNNNetInstance.Session mSession;
    private MNNNetInstance.Session.Tensor mInputTensor;
    private MNNNetInstance.Session.Tensor mOutputTensor;
    
    // 解码器相关
    private MNNNetInstance mDecoderNetInstance;
    private MNNNetInstance.Session mDecoderSession;
    private MNNNetInstance.Session.Tensor mDecoderInputTensor;
    private MNNNetInstance.Session.Tensor mDecoderOutputTensor;
    
    private String mModelPath;
    private String mDecoderModelPath;
    private String mWavPath;
    private float[] mAudioData;
    private int[] mCodesData; // 导入的编码数据
    private int[] mCodesShape; // 导入编码的形状
    private float[] mDecodedAudioData; // 解码后的音频数据
    private String mLastResultFilePath; // 最近的编码结果文件路径
    private String mLastCodesFilePath; // 最近导入的编码文件路径
    private AudioPlayer mAudioPlayer;
    private int mSelectedQuantizerLayers = 32; // 默认为32

    private class ModelPrepareTask extends AsyncTask<String, Void, String> {
        @Override
        protected String doInBackground(String... params) {
            return prepareModel();
        }

        @Override
        protected void onPostExecute(String result) {
            if ("success".equals(result)) {
                mStatusText.setText("模型准备完成，点击开始编码");
                mEncodeButton.setEnabled(true);
            } else {
                mStatusText.setText("模型准备失败: " + result);
                Log.e(TAG, "模型准备失败: " + result);
            }
        }
    }

    private class AudioEncodeResult {
        public int[] codes;
        public float inferenceTime;
        public float realTimeRatio;
        public float audioDuration;
        public String errorMessage;
        public int[] outputShape;
    }

    private class AudioDecodeResult {
        public float[] waveform;
        public float inferenceTime;
        public float realTimeRatio;
        public float audioDuration;
        public String errorMessage;
        public int[] outputShape;
    }

    private class AudioEncodeTask extends AsyncTask<String, Void, AudioEncodeResult> {
        @Override
        protected AudioEncodeResult doInBackground(String... params) {
            return performAudioEncode();
        }

        @Override
        protected void onPostExecute(AudioEncodeResult result) {
            if (result.errorMessage != null) {
                mResultText.setText("编码失败: " + result.errorMessage);
                mTimeText.setText("");
            } else {
                // 根据用户选择的层数对结果进行切片
                int[] fullCodes = result.codes;
                int[] fullShape = result.outputShape;
                int batch = fullShape[1];
                int frames = fullShape[2];

                int[] selectedCodes = new int[mSelectedQuantizerLayers * batch * frames];
                System.arraycopy(fullCodes, 0, selectedCodes, 0, selectedCodes.length);
                
                int[] selectedShape = {mSelectedQuantizerLayers, batch, frames};

                // 使用切片后的数据更新结果对象
                result.codes = selectedCodes;
                result.outputShape = selectedShape;

                StringBuilder sb = new StringBuilder();
                sb.append("编码成功!\n\n");
                
                // 性能信息
                sb.append("=== 性能指标 ===\n");
                sb.append("音频时长: ").append(String.format("%.2f", result.audioDuration)).append(" 秒\n");
                sb.append("编码时间: ").append(String.format("%.2f", result.inferenceTime)).append(" ms\n");
                sb.append("实时率: ").append(String.format("%.2f", result.realTimeRatio)).append("x\n\n");
                
                // 编码信息
                sb.append("=== 编码结果 ===\n");
                sb.append("输出形状: [").append(result.outputShape[0])
                  .append(", ").append(result.outputShape[1])
                  .append(", ").append(result.outputShape[2]).append("]\n");
                sb.append("编码数量: ").append(result.codes.length).append("\n");
                sb.append("编码范围: [").append(getArrayMin(result.codes))
                  .append(", ").append(getArrayMax(result.codes)).append("]\n");
                
                // 计算音频码率
                int quantizerLayers = result.outputShape[0]; // 第一维是量化器层数
                float hopLength = 640.0f; // FunCodec的hop_length
                float sampleRate = 16000.0f;
                float bitsPerQuantizer = (float)(Math.log(1024) / Math.log(2)); // log2(codebook_size)
                float bitrate = (quantizerLayers * bitsPerQuantizer * sampleRate) / hopLength;
                
                sb.append("量化器层数: ").append(quantizerLayers).append("/32\n");
                sb.append("音频码率: ").append(String.format("%.1f", bitrate / 1000.0f)).append(" kbps\n");
                sb.append("压缩比: ").append(String.format("%.1f", 
                    (float)mAudioData.length / result.codes.length)).append(":1\n\n");
                sb.append("前20个编码: ");
                for (int i = 0; i < Math.min(20, result.codes.length); i++) {
                    sb.append(result.codes[i]).append(" ");
                }
                sb.append("\n");
                
                mResultText.setText(sb.toString());
                
                // 在时间显示区域显示关键性能指标
                mTimeText.setText(String.format("编码时间: %.2f ms  |  实时率: %.2fx", 
                    result.inferenceTime, result.realTimeRatio));
                
                // 保存编码结果
                saveEncodingResults(result.codes, result.outputShape);
            }
        }
    }

    private class AudioDecodeTask extends AsyncTask<String, Void, AudioDecodeResult> {
        @Override
        protected AudioDecodeResult doInBackground(String... params) {
            return performAudioDecode();
        }

        @Override
        protected void onPostExecute(AudioDecodeResult result) {
            if (result.errorMessage != null) {
                mResultText.setText("解码失败: " + result.errorMessage);
                mTimeText.setText("");
            } else {
                StringBuilder sb = new StringBuilder();
                sb.append("解码成功!\n\n");
                
                // 性能信息
                sb.append("=== 性能指标 ===\n");
                sb.append("解码时间: ").append(String.format("%.2f", result.inferenceTime)).append(" ms\n");
                sb.append("音频时长: ").append(String.format("%.2f", result.audioDuration)).append(" 秒\n");
                sb.append("实时率: ").append(String.format("%.2f", result.realTimeRatio)).append("x\n\n");
                
                // 解码信息
                sb.append("=== 解码结果 ===\n");
                sb.append("输出形状: [").append(result.outputShape[0])
                  .append(", ").append(result.outputShape[1])
                  .append(", ").append(result.outputShape[2]).append("]\n");
                sb.append("音频样本数: ").append(result.waveform.length).append("\n");
                sb.append("音频范围: [").append(String.format("%.3f", getArrayMinFloat(result.waveform)))
                  .append(", ").append(String.format("%.3f", getArrayMaxFloat(result.waveform))).append("]\n");
                
                // 新增：计算输出音频码率 (PCM)
                float outputSampleRate = 16000.0f;
                int outputBitDepth = 16; // 保存为WAV时通常是16-bit
                int outputChannels = result.outputShape.length > 1 ? result.outputShape[1] : 1; // 形状为 [B, C, T]
                float outputBitrate = outputSampleRate * outputBitDepth * outputChannels;
                sb.append("输出码率 (PCM): ").append(String.format("%.0f", outputBitrate / 1000.0f)).append(" kbps\n");

                // 计算解码音频的码率信息
                if (mCodesData != null && mCodesShape != null && mCodesShape.length == 3) {
                    int quantizerLayers = mCodesShape[0]; // 使用真实的量化层数
                    float hopLength = 640.0f;
                    float sampleRate = 16000.0f;
                    float bitsPerQuantizer = (float)(Math.log(1024) / Math.log(2));
                    float bitrate = (quantizerLayers * bitsPerQuantizer * sampleRate) / hopLength;
                    sb.append("输入码率: ").append(String.format("%.1f", bitrate / 1000.0f)).append(" kbps\n");
                    sb.append("量化器层数: ").append(quantizerLayers).append("\n");
                }
                sb.append("\n");
                
                mResultText.setText(sb.toString());
                
                // 在时间显示区域显示关键性能指标
                mTimeText.setText(String.format("解码时间: %.2f ms  |  音频时长: %.2f秒", 
                    result.inferenceTime, result.audioDuration));
                
                // 启用音频控制按钮
                mPlayAudioButton.setEnabled(true);
                mSaveAudioButton.setEnabled(true);
                
                // 保存解码后的音频数据
                mDecodedAudioData = result.waveform;
            }
        }
    }



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_audio_codec);

        initViews();
        
        mStatusText.setText("申请存储权限...");
        mEncodeButton.setEnabled(false);
        
        // 请求存储权限
        requestStoragePermission();
    }
    
    private void requestStoragePermission() {
        // 使用MNN demo的标准权限请求方式
        PermissionUtils.askPermission(this, new String[]{
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        }, 100, new Runnable() {
            @Override
            public void run() {
                startModelPreparation();
            }
        });
    }
    
    private void startModelPreparation() {
        mStatusText.setText("正在准备模型...");
        new ModelPrepareTask().execute();
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        
        if (requestCode == 100) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // 权限授予，开始模型准备 (PermissionUtils会自动调用Runnable)
                Log.d(TAG, "存储权限已授予");
            } else {
                // 权限被拒绝
                mStatusText.setText("需要存储权限才能加载模型文件。请在设置中授予权限后重新打开应用。");
                Toast.makeText(this, "没有获得必要的存储权限", Toast.LENGTH_LONG).show();
            }
        }
    }

    private void initViews() {
        mStatusText = findViewById(R.id.statusText);
        mResultText = findViewById(R.id.resultText);
        mTimeText = findViewById(R.id.timeText);
        mEncodeButton = findViewById(R.id.encodeButton);
        mImportWavButton = findViewById(R.id.importWavButton);
        mShareResultButton = findViewById(R.id.shareResultButton);
        mImportCodesButton = findViewById(R.id.importCodesButton);
        mDecodeButton = findViewById(R.id.decodeButton);
        mPlayAudioButton = findViewById(R.id.playAudioButton);
        mSaveAudioButton = findViewById(R.id.saveAudioButton);
        
        mQuantizerSeekBar = findViewById(R.id.quantizerSeekBar);
        mQuantizerValueText = findViewById(R.id.quantizerValueText);
        
        mEncodeButton.setOnClickListener(this);
        mImportWavButton.setOnClickListener(this);
        mShareResultButton.setOnClickListener(this);
        mImportCodesButton.setOnClickListener(this);
        mDecodeButton.setOnClickListener(this);
        mPlayAudioButton.setOnClickListener(this);
        mSaveAudioButton.setOnClickListener(this);
        
        // 设置SeekBar
        mQuantizerSeekBar.setMax(31); // 对应 1-32 层
        mQuantizerSeekBar.setProgress(31); // 默认32层
        mQuantizerValueText.setText(String.format("%d / 32", mSelectedQuantizerLayers));
        mQuantizerSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                mSelectedQuantizerLayers = progress + 1;
                mQuantizerValueText.setText(String.format("%d / 32", mSelectedQuantizerLayers));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                // 无需处理
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                // 无需处理
            }
        });

        // 初始状态下某些按钮不可用
        mShareResultButton.setEnabled(false);
        mDecodeButton.setEnabled(false);
        mPlayAudioButton.setEnabled(false);
        mSaveAudioButton.setEnabled(false);
        
        // 初始化音频播放器
        mAudioPlayer = new AudioPlayer();
    }

    private String prepareModel() {
        try {
            // 准备编码器模型文件 - 采用MNN标准方式
            mModelPath = getCacheDir() + "/funcodec_encoder.mnn";
            File cacheModel = new File(mModelPath);
            
            // 准备解码器模型文件
            mDecoderModelPath = getCacheDir() + "/funcodec_decoder.mnn";
            File cacheDecoderModel = new File(mDecoderModelPath);
            
            // 检查缓存中是否已有有效的编码器模型文件
            if (cacheModel.exists() && cacheModel.length() > 100000000) { // 100MB+
                Log.d(TAG, "使用缓存中的编码器模型文件: " + mModelPath + ", 大小: " + cacheModel.length());
            } else {
                Log.d(TAG, "缓存中无有效编码器模型文件，尝试复制");
                
                // 优先尝试从assets复制（遵循MNN标准方式）
                boolean assetsSuccess = false;
                try {
                    Log.d(TAG, "尝试从assets复制编码器模型文件: " + MODEL_FILE_NAME);
                    
                    // 使用改进的大文件复制方法
                    assetsSuccess = copyLargeAssetFile(MODEL_FILE_NAME, mModelPath);
                    
                    if (assetsSuccess) {
                        Log.d(TAG, "编码器从assets复制成功, 大小: " + cacheModel.length());
                    } else {
                        Log.w(TAG, "编码器从assets复制失败");
                    }
                } catch (Exception e) {
                    Log.w(TAG, "编码器从assets复制异常: " + e.getMessage());
                    e.printStackTrace();
                }
                
                // 如果assets复制失败，尝试外部存储
                if (!assetsSuccess) {
                    String externalPath = Environment.getExternalStorageDirectory() + "/MNNDemo/funcodec_encoder.mnn";
                    File externalModel = new File(externalPath);
                    
                    if (externalModel.exists()) {
                        Log.d(TAG, "从外部存储复制编码器模型: " + externalPath + ", 大小: " + externalModel.length());
                        if (copyFileToCache(externalPath, mModelPath)) {
                            Log.d(TAG, "编码器外部存储复制成功");
                        } else {
                            // 如果复制失败，直接使用外部路径
                            Log.d(TAG, "编码器复制失败，直接使用外部存储路径");
                            mModelPath = externalPath;
                        }
                    } else {
                        return "编码器模型文件未找到。请确保funcodec_encoder.mnn文件在: " + externalPath;
                    }
                }
            }

            // 检查缓存中是否已有有效的解码器模型文件
            if (cacheDecoderModel.exists() && cacheDecoderModel.length() > 1000000) { // 1MB+
                Log.d(TAG, "使用缓存中的解码器模型文件: " + mDecoderModelPath + ", 大小: " + cacheDecoderModel.length());
            } else {
                Log.d(TAG, "缓存中无有效解码器模型文件，尝试复制");
                
                // 尝试从assets复制解码器模型
                boolean decoderAssetsSuccess = false;
                try {
                    Log.d(TAG, "尝试从assets复制解码器模型文件: " + DECODER_MODEL_FILE_NAME);
                    
                    // 使用改进的大文件复制方法
                    decoderAssetsSuccess = copyLargeAssetFile(DECODER_MODEL_FILE_NAME, mDecoderModelPath);
                    
                    if (decoderAssetsSuccess) {
                        Log.d(TAG, "解码器从assets复制成功, 大小: " + cacheDecoderModel.length());
                    } else {
                        Log.w(TAG, "解码器从assets复制失败");
                    }
                } catch (Exception e) {
                    Log.w(TAG, "解码器从assets复制异常: " + e.getMessage());
                    e.printStackTrace();
                }
                
                // 如果assets复制失败，尝试外部存储
                if (!decoderAssetsSuccess) {
                    String externalDecoderPath = Environment.getExternalStorageDirectory() + "/MNNDemo/funcodec_decoder.mnn";
                    File externalDecoderModel = new File(externalDecoderPath);
                    
                    if (externalDecoderModel.exists()) {
                        Log.d(TAG, "从外部存储复制解码器模型: " + externalDecoderPath + ", 大小: " + externalDecoderModel.length());
                        if (copyFileToCache(externalDecoderPath, mDecoderModelPath)) {
                            Log.d(TAG, "解码器外部存储复制成功");
                        } else {
                            // 如果复制失败，直接使用外部路径
                            Log.d(TAG, "解码器复制失败，直接使用外部存储路径");
                            mDecoderModelPath = externalDecoderPath;
                        }
                    } else {
                        Log.w(TAG, "解码器模型文件未找到。assets和外部存储中都没有找到 funcodec_decoder.mnn");
                        // 解码器是可选的，不影响编码功能
                    }
                }
            }

            // 准备音频文件
            mWavPath = getCacheDir() + "/example.wav";
            try {
                Common.copyAssetResource2File(this, EXAMPLE_WAV_NAME, mWavPath);
            } catch (Exception e) {
                // 如果assets中没有example.wav，则生成一个示例文件
                Log.d(TAG, "assets中未找到音频文件，生成示例音频文件");
                if (!WavFileGenerator.generateExampleWav(AudioCodecActivity.this, "example.wav")) {
                    return "示例音频文件生成失败";
                }
            }

            // 加载音频数据
            mAudioData = AudioFileReader.loadWavFile(mWavPath, 16000);
            if (mAudioData == null) {
                return "音频文件加载失败";
            }

            // 验证编码器模型文件
            File finalModelFile = new File(mModelPath);
            if (!finalModelFile.exists()) {
                return "编码器模型文件不存在: " + mModelPath;
            }
            if (finalModelFile.length() == 0) {
                return "编码器模型文件为空: " + mModelPath;
            }
            
            Log.d(TAG, "最终编码器模型文件: " + mModelPath + ", 大小: " + finalModelFile.length() + " 字节");
            Log.d(TAG, "编码器文件可读: " + finalModelFile.canRead());
            
            // 创建编码器MNN实例 - 使用MNN标准方式
            Log.d(TAG, "开始创建编码器MNN实例...");
            mNetInstance = MNNNetInstance.createFromFile(mModelPath);
            
            if (mNetInstance == null) {
                return "编码器MNN实例创建失败。模型文件: " + mModelPath + " (大小: " + finalModelFile.length() + ")。请检查模型文件是否为有效的MNN格式。";
            }
            
            Log.d(TAG, "编码器MNN实例创建成功");

            // 创建编码器Session - 使用标准配置
            Log.d(TAG, "开始创建编码器MNN Session...");
            MNNNetInstance.Config config = new MNNNetInstance.Config();
            config.numThread = 4;
            config.forwardType = MNNForwardType.FORWARD_CPU.type;
            
            mSession = mNetInstance.createSession(config);
            if (mSession == null) {
                return "编码器MNN Session创建失败。请检查模型兼容性。";
            }
            Log.d(TAG, "编码器MNN Session创建成功");

            // 获取编码器输入输出tensor
            Log.d(TAG, "获取编码器输入输出tensor...");
            mInputTensor = mSession.getInput(null);
            mOutputTensor = mSession.getOutput(null);
            
            if (mInputTensor == null) {
                return "编码器输入tensor获取失败。模型可能没有输入层或输入层名称不匹配。";
            }
            if (mOutputTensor == null) {
                return "编码器输出tensor获取失败。模型可能没有输出层或输出层名称不匹配。";
            }
            
            Log.d(TAG, "编码器输入输出tensor获取成功");

            Log.d(TAG, "音频数据长度: " + mAudioData.length);
            Log.d(TAG, "编码器输入tensor维度: " + java.util.Arrays.toString(mInputTensor.getDimensions()));
            Log.d(TAG, "编码器输出tensor维度: " + java.util.Arrays.toString(mOutputTensor.getDimensions()));

            // 创建解码器MNN实例（如果解码器模型存在）- 使用优化配置
            File finalDecoderModelFile = new File(mDecoderModelPath);
            if (finalDecoderModelFile.exists() && finalDecoderModelFile.length() > 1000000) { // 至少1MB
                Log.d(TAG, "开始创建解码器MNN实例...");
                Log.d(TAG, "解码器模型文件: " + mDecoderModelPath + ", 大小: " + finalDecoderModelFile.length() + " 字节");
                
                try {
                    mDecoderNetInstance = MNNNetInstance.createFromFile(mDecoderModelPath);
                    if (mDecoderNetInstance != null) {
                        Log.d(TAG, "解码器MNN实例创建成功");
                        
                        // 解码器使用优化配置 - 减少线程数避免CPU过载
                        MNNNetInstance.Config decoderConfig = new MNNNetInstance.Config();
                        decoderConfig.numThread = 1; 
                        decoderConfig.forwardType = MNNForwardType.FORWARD_CPU.type;
                        
                        Log.d(TAG, "解码器配置: 线程数=" + decoderConfig.numThread + ", 推理类型=CPU");
                        
                        mDecoderSession = mDecoderNetInstance.createSession(decoderConfig);
                        if (mDecoderSession != null) {
                            Log.d(TAG, "解码器Session创建成功");
                            
                            mDecoderInputTensor = mDecoderSession.getInput(null);
                            mDecoderOutputTensor = mDecoderSession.getOutput(null);
                            
                            if (mDecoderInputTensor != null && mDecoderOutputTensor != null) {
                                Log.d(TAG, "解码器tensor获取成功");
                                Log.d(TAG, "解码器输入tensor维度: " + java.util.Arrays.toString(mDecoderInputTensor.getDimensions()));
                                Log.d(TAG, "解码器输出tensor维度: " + java.util.Arrays.toString(mDecoderOutputTensor.getDimensions()));
                                Log.d(TAG, "解码器初始化完成");
                            } else {
                                Log.w(TAG, "解码器tensor获取失败");
                                mDecoderSession = null;
                            }
                        } else {
                            Log.w(TAG, "解码器Session创建失败");
                            mDecoderNetInstance.release();
                            mDecoderNetInstance = null;
                        }
                    } else {
                        Log.w(TAG, "解码器MNN实例创建失败");
                    }
                } catch (Exception e) {
                    Log.w(TAG, "解码器初始化异常: " + e.getMessage());
                    e.printStackTrace();
                    // 清理资源
                    if (mDecoderNetInstance != null) {
                        mDecoderNetInstance.release();
                        mDecoderNetInstance = null;
                    }
                }
            } else {
                Log.d(TAG, "解码器模型文件不存在或过小，跳过解码器初始化。文件路径: " + mDecoderModelPath);
                if (finalDecoderModelFile.exists()) {
                    Log.d(TAG, "解码器文件大小: " + finalDecoderModelFile.length() + " 字节 (需要至少1MB)");
                }
            }

            return "success";
            
        } catch (Exception e) {
            Log.e(TAG, "模型准备异常", e);
            return "异常: " + e.getMessage();
        }
    }

    private AudioEncodeResult performAudioEncode() {
        AudioEncodeResult result = new AudioEncodeResult();
        
        try {
            Log.d(TAG, "开始音频编码...");
            Log.d(TAG, "音频数据长度: " + mAudioData.length);
            
            // 关键修复：检查tensor是否为null，如果是则重新初始化
            if (mInputTensor == null || mOutputTensor == null || mSession == null) {
                Log.w(TAG, "Tensor或Session为null，尝试重新初始化模型...");
                String reinitResult = reinitializeModel();
                if (!"success".equals(reinitResult)) {
                    result.errorMessage = "模型重新初始化失败: " + reinitResult;
                    return result;
                }
            }
            
            // 再次检查tensor状态
            if (mInputTensor == null) {
                result.errorMessage = "输入tensor为null，无法进行编码";
                return result;
            }
            if (mOutputTensor == null) {
                result.errorMessage = "输出tensor为null，无法进行编码";
                return result;
            }
            
            Log.d(TAG, "输入tensor原始维度: " + java.util.Arrays.toString(mInputTensor.getDimensions()));
            
            // 调整输入tensor形状 - 重要：先reshape再设置数据
            int[] inputShape = {1, mAudioData.length};
            Log.d(TAG, "调整输入tensor形状为: " + java.util.Arrays.toString(inputShape));
            mInputTensor.reshape(inputShape);
            
            // 重要：reshape tensor后需要调用session的reshape方法
            Log.d(TAG, "调用session.reshape()...");
            mSession.reshape();
            
            // 验证reshape结果
            int[] actualShape = mInputTensor.getDimensions();
            Log.d(TAG, "reshape后的实际形状: " + java.util.Arrays.toString(actualShape));
            
            // 关键修复：在reshape后重新获取输出tensor，确保维度正确更新
            mOutputTensor = mSession.getOutput(null);
            if (mOutputTensor == null) {
                result.errorMessage = "reshape后输出tensor获取失败";
                return result;
            }
            
            // 检查输出tensor是否也正确初始化了
            int[] outputShape = mOutputTensor.getDimensions();
            Log.d(TAG, "输出tensor形状: " + java.util.Arrays.toString(outputShape));
            
            // 设置输入数据 - 使用更安全的方式
            Log.d(TAG, "设置输入数据...");
            float[] inputData = new float[mAudioData.length];
            System.arraycopy(mAudioData, 0, inputData, 0, mAudioData.length);
            
            // 尝试使用setInputFloatData
            try {
                mInputTensor.setInputFloatData(inputData);
                Log.d(TAG, "输入数据设置成功");
            } catch (Exception e) {
                Log.e(TAG, "setInputFloatData失败: " + e.getMessage());
                result.errorMessage = "输入数据设置失败: " + e.getMessage();
                return result;
            }
            
            // 执行推理
            Log.d(TAG, "开始推理...");
            long startTime = System.nanoTime();
            mSession.run();
            long endTime = System.nanoTime();
            
            result.inferenceTime = (endTime - startTime) / 1000000.0f;
            
            // 计算音频时长和实时率
            result.audioDuration = mAudioData.length / 16000.0f; // 16kHz采样率
            result.realTimeRatio = result.audioDuration * 1000 / result.inferenceTime;
            
            Log.d(TAG, "推理完成，耗时: " + result.inferenceTime + "ms");
            Log.d(TAG, "音频时长: " + result.audioDuration + "秒");
            Log.d(TAG, "实时率: " + result.realTimeRatio + "x");
            
            // 获取输出tensor信息
            Log.d(TAG, "获取输出结果...");
            Log.d(TAG, "输出tensor维度: " + java.util.Arrays.toString(mOutputTensor.getDimensions()));
            
            // 获取输出数据
            result.codes = mOutputTensor.getIntData();
            result.outputShape = mOutputTensor.getDimensions();
            
            Log.d(TAG, "输出形状: " + java.util.Arrays.toString(result.outputShape));
            Log.d(TAG, "输出数据长度: " + (result.codes != null ? result.codes.length : "null"));
            
            // 验证输出数据的一致性
            if (result.codes != null && result.outputShape != null) {
                int expectedLength = 1;
                for (int dim : result.outputShape) {
                    expectedLength *= dim;
                }
                if (result.codes.length != expectedLength) {
                    Log.w(TAG, "警告：输出数据长度(" + result.codes.length + 
                        ")与形状计算的长度(" + expectedLength + ")不匹配！");
                }
            }
            
        } catch (Exception e) {
            Log.e(TAG, "音频编码异常", e);
            result.errorMessage = "编码异常: " + e.getMessage();
            e.printStackTrace();
        }
        
        return result;
    }

    /**
     * 重新初始化模型（当tensor为null时调用）
     */
    private String reinitializeModel() {
        try {
            Log.d(TAG, "开始重新初始化编码器模型...");
            
            // 清理旧的资源
            if (mSession != null) {
                mSession.release();
            }
            if (mNetInstance != null) {
                mNetInstance.release();
            }
            
            // 重新创建MNN实例
            File modelFile = new File(mModelPath);
            if (!modelFile.exists()) {
                return "模型文件不存在: " + mModelPath;
            }
            
            mNetInstance = MNNNetInstance.createFromFile(mModelPath);
            if (mNetInstance == null) {
                return "MNN实例创建失败";
            }
            
            // 重新创建Session
            MNNNetInstance.Config config = new MNNNetInstance.Config();
            config.numThread = 4;
            config.forwardType = MNNForwardType.FORWARD_CPU.type;
            
            mSession = mNetInstance.createSession(config);
            if (mSession == null) {
                return "Session创建失败";
            }
            
            // 重新获取tensor
            mInputTensor = mSession.getInput(null);
            mOutputTensor = mSession.getOutput(null);
            
            if (mInputTensor == null) {
                return "输入tensor获取失败";
            }
            if (mOutputTensor == null) {
                return "输出tensor获取失败";
            }
            
            // 确保tensor状态清理
            Log.d(TAG, "重新初始化后清理tensor状态...");
            
            Log.d(TAG, "模型重新初始化成功");
            Log.d(TAG, "输入tensor维度: " + java.util.Arrays.toString(mInputTensor.getDimensions()));
            Log.d(TAG, "输出tensor维度: " + java.util.Arrays.toString(mOutputTensor.getDimensions()));
            
            return "success";
            
        } catch (Exception e) {
            Log.e(TAG, "模型重新初始化失败", e);
            return "重新初始化异常: " + e.getMessage();
        }
    }

    private AudioDecodeResult performAudioDecode() {
        AudioDecodeResult result = new AudioDecodeResult();
        
        try {
            if (mDecoderSession == null || mCodesData == null) {
                result.errorMessage = "解码器未准备好或编码数据为空";
                return result;
            }

            if (mCodesShape == null || mCodesShape.length != 3) {
                result.errorMessage = "编码形状信息无效或缺失";
                return result;
            }
            
            Log.d(TAG, "开始音频解码...");
            Log.d(TAG, "编码数据长度: " + mCodesData.length);
            Log.d(TAG, "编码形状 (来自编码器): " + java.util.Arrays.toString(mCodesShape));

            // MNN解码器期望的输入形状是 [batch, layers, frames]
            // 编码器输出的形状是 [layers, batch, frames]
            // 因此需要进行转置
            int layers = mCodesShape[0];
            int batch = mCodesShape[1];
            int totalFrames = mCodesShape[2];
            final int DECODER_EXPECTED_LAYERS = 32;

            // 验证数据长度和形状是否匹配
            int expectedLength = layers * batch * totalFrames;
            if (mCodesData.length != expectedLength) {
                result.errorMessage = "编码数据长度(" + mCodesData.length + ")与形状 "
                        + java.util.Arrays.toString(mCodesShape) + " 不匹配 (应为 " + expectedLength + ")";
                return result;
            }

            // 对编码数据进行转置 [layers, batch, frames] -> [batch, layers, frames]
            int[] transposedCodesData = transpose(mCodesData, new int[]{0, 1, 2}, new int[]{1, 0, 2}, mCodesShape);
            
            // 如果实际层数小于解码器期望的32层，进行0填充
            int[] finalPaddedCodesData;
            if (layers < DECODER_EXPECTED_LAYERS) {
                Log.d(TAG, "将编码从 " + layers + " 层填充到 " + DECODER_EXPECTED_LAYERS + " 层");
                finalPaddedCodesData = new int[batch * DECODER_EXPECTED_LAYERS * totalFrames];
                // Java数组默认初始化为0，我们只需要将有数据的部分复制过去
                // 因为batch=1，所以可以直接复制
                System.arraycopy(transposedCodesData, 0, finalPaddedCodesData, 0, transposedCodesData.length);
            } else {
                finalPaddedCodesData = transposedCodesData;
            }

            int[] finalInputShape;
            int[] finalCodesData;
            int[] decoderInputShape = {batch, DECODER_EXPECTED_LAYERS, totalFrames};

            // 检查输入数据大小，如果过大则分块处理以避免内存溢出
            if (totalFrames > 500) {
                Log.w(TAG, "输入数据过大 (" + totalFrames + " 帧)，进行分块处理");
                int limitedFrames = 500;
                Log.d(TAG, "限制处理长度为: " + limitedFrames + " 帧");

                finalInputShape = new int[]{batch, DECODER_EXPECTED_LAYERS, limitedFrames};
                
                int limitedCodesDataLength = batch * DECODER_EXPECTED_LAYERS * limitedFrames;
                finalCodesData = new int[limitedCodesDataLength];
                // 从已经填充和转置的数据中截取
                System.arraycopy(finalPaddedCodesData, 0, finalCodesData, 0, limitedCodesDataLength);
            } else {
                finalInputShape = decoderInputShape;
                finalCodesData = finalPaddedCodesData;
            }

            Log.d(TAG, "调整解码器输入tensor形状为: " + java.util.Arrays.toString(finalInputShape));
            mDecoderInputTensor.reshape(finalInputShape);
            mDecoderSession.reshape();
            
            // 关键修复：在reshape后重新获取输出tensor，确保维度正确更新
            mDecoderOutputTensor = mDecoderSession.getOutput(null);
            if (mDecoderOutputTensor == null) {
                result.errorMessage = "解码器reshape后输出tensor获取失败";
                return result;
            }
            
            // 验证解码器输出tensor形状
            int[] decoderOutputShape = mDecoderOutputTensor.getDimensions();
            Log.d(TAG, "解码器输出tensor形状: " + java.util.Arrays.toString(decoderOutputShape));
            
            // 设置输入数据
            Log.d(TAG, "设置编码数据...");
            mDecoderInputTensor.setInputIntData(finalCodesData);
            
            // 执行推理 - 添加超时检查
            Log.d(TAG, "开始解码推理...");
            long startTime = System.nanoTime();
            
            // 使用线程池执行推理，避免阻塞主线程
            final boolean[] inferenceComplete = {false};
            final Exception[] inferenceException = {null};
            
            Thread inferenceThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        mDecoderSession.run();
                        inferenceComplete[0] = true;
                    } catch (Exception e) {
                        inferenceException[0] = e;
                    }
                }
            });
            
            inferenceThread.start();
            
            // 等待推理完成，但设置超时
            int timeoutSeconds = 300; // 30秒超时
            int checkIntervalMs = 500; // 每500ms检查一次
            int totalWaitTime = 0;
            
            while (!inferenceComplete[0] && inferenceException[0] == null && totalWaitTime < timeoutSeconds * 1000) {
                try {
                    Thread.sleep(checkIntervalMs);
                    totalWaitTime += checkIntervalMs;
                    
                    // 每5秒打印一次进度
                    if (totalWaitTime % 5000 == 0) {
                        Log.d(TAG, "解码进行中... 已等待 " + (totalWaitTime / 1000) + " 秒");
                    }
                } catch (InterruptedException e) {
                    Log.w(TAG, "解码等待被中断");
                    break;
                }
            }
            
            if (inferenceException[0] != null) {
                throw inferenceException[0];
            }
            
            if (!inferenceComplete[0]) {
                // 超时处理
                Log.e(TAG, "解码推理超时 (" + timeoutSeconds + "秒)");
                result.errorMessage = "解码推理超时，模型可能过于复杂或输入数据过大";
                
                // 尝试中断推理线程
                if (inferenceThread.isAlive()) {
                    inferenceThread.interrupt();
                }
                
                return result;
            }
            
            long endTime = System.nanoTime();
            result.inferenceTime = (endTime - startTime) / 1000000.0f;
            
            // 获取输出数据
            Log.d(TAG, "获取解码结果...");
            result.waveform = mDecoderOutputTensor.getFloatData();
            result.outputShape = mDecoderOutputTensor.getDimensions();
            
            // 验证输出数据
            if (result.waveform == null || result.waveform.length == 0) {
                result.errorMessage = "解码输出为空";
                return result;
            }
            
            // 新增：归一化音频数据，防止削波失真
            result.waveform = normalizeFloatArray(result.waveform);
            
            // 计算音频时长和实时率
            result.audioDuration = result.waveform.length / 16000.0f; // 16kHz采样率
            result.realTimeRatio = result.audioDuration * 1000 / result.inferenceTime;
            
            // 保存解码结果供播放使用
            mDecodedAudioData = result.waveform;
            
            Log.d(TAG, "解码完成，耗时: " + result.inferenceTime + "ms");
            Log.d(TAG, "音频时长: " + result.audioDuration + "秒");
            Log.d(TAG, "实时率: " + result.realTimeRatio + "x");
            Log.d(TAG, "输出形状: " + java.util.Arrays.toString(result.outputShape));
            Log.d(TAG, "输出数据长度: " + result.waveform.length);
            
            // 启用播放和保存按钮
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    mPlayAudioButton.setEnabled(true);
                    mSaveAudioButton.setEnabled(true);
                }
            });
            
        } catch (Exception e) {
            Log.e(TAG, "音频解码异常", e);
            result.errorMessage = "解码异常: " + e.getMessage();
            e.printStackTrace();
        }
        
        return result;
    }



    private void saveEncodingResults(int[] codes, int[] outputShape) {
        try {
            // 使用应用私有目录，避免权限问题
            File outputDir = new File(getExternalFilesDir(null), "results");
            if (!outputDir.exists()) {
                outputDir.mkdirs();
            }
            
            // 生成带时间戳的文件名
            String timestamp = String.valueOf(System.currentTimeMillis());
            String fileName = "audio_encoding_result_" + timestamp + ".txt";
            File outputFile = new File(outputDir, fileName);
            
            FileOutputStream fos = new FileOutputStream(outputFile);
            
            StringBuilder sb = new StringBuilder();
            sb.append("# 音频编码结果\n");
            sb.append("# 时间戳: ").append(timestamp).append("\n");
            sb.append("# 输出形状: [").append(outputShape[0])
              .append(", ").append(outputShape[1])
              .append(", ").append(outputShape[2]).append("]\n");
            // 添加可机读的形状信息
            sb.append("shape:").append(outputShape[0]).append(",").append(outputShape[1]).append(",").append(outputShape[2]).append("\n");
            sb.append("# 编码数量: ").append(codes.length).append("\n");
            sb.append("# 音频时长: ").append(mAudioData.length / 16000.0f).append(" 秒\n");
            sb.append("# 编码数据:\n");
            
            for (int code : codes) {
                sb.append(code).append("\n");
            }
            
            fos.write(sb.toString().getBytes());
            fos.close();
            
            // 保存文件路径供分享使用
            mLastResultFilePath = outputFile.getAbsolutePath();
            
            // 启用分享按钮
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    mShareResultButton.setEnabled(true);
                }
            });
            
            Log.d(TAG, "编码结果已保存到: " + outputFile.getAbsolutePath());
            Toast.makeText(this, "编码结果已保存到应用目录", Toast.LENGTH_SHORT).show();
            
        } catch (IOException e) {
            Log.e(TAG, "保存编码结果失败", e);
            Toast.makeText(this, "保存编码结果失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }


    
    private void copyStream(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[8192];
        int bytesRead;
        while ((bytesRead = in.read(buffer)) != -1) {
            out.write(buffer, 0, bytesRead);
        }
    }
    
    private boolean copyFileToCache(String sourcePath, String targetPath) {
        try {
            InputStream in = new FileInputStream(sourcePath);
            FileOutputStream out = new FileOutputStream(targetPath);
            
            copyStream(in, out);
            
            in.close();
            out.close();
            
            // 验证复制结果
            File targetFile = new File(targetPath);
            File sourceFile = new File(sourcePath);
            
            if (targetFile.exists() && targetFile.length() == sourceFile.length()) {
                Log.d(TAG, "文件复制成功: " + targetPath + ", 大小: " + targetFile.length());
                return true;
            } else {
                Log.e(TAG, "文件复制验证失败");
                return false;
            }
            
        } catch (IOException e) {
            Log.e(TAG, "文件复制失败: " + sourcePath + " -> " + targetPath, e);
            return false;
        }
    }

    private int getArrayMin(int[] array) {
        int min = Integer.MAX_VALUE;
        for (int value : array) {
            min = Math.min(min, value);
        }
        return min;
    }

    private int getArrayMax(int[] array) {
        int max = Integer.MIN_VALUE;
        for (int value : array) {
            max = Math.max(max, value);
        }
        return max;
    }

    private float getArrayMinFloat(float[] array) {
        float min = Float.MAX_VALUE;
        for (float value : array) {
            min = Math.min(min, value);
        }
        return min;
    }

    private float getArrayMaxFloat(float[] array) {
        float max = Float.MIN_VALUE;
        for (float value : array) {
            max = Math.max(max, value);
        }
        return max;
    }

    /**
     * 对存储在一维数组中的3D张量数据进行转置
     * @param data 原始一维数据
     * @param srcAxes 原始维度顺序 (e.g., {0, 1, 2} for layers, batch, frames)
     * @param dstAxes 目标维度顺序 (e.g., {1, 0, 2} for batch, layers, frames)
     * @param shape 原始形状
     * @return 转置后的一维数据
     */
    private int[] transpose(int[] data, int[] srcAxes, int[] dstAxes, int[] shape) {
        int[] transposedShape = new int[3];
        transposedShape[0] = shape[dstAxes[0]];
        transposedShape[1] = shape[dstAxes[1]];
        transposedShape[2] = shape[dstAxes[2]];

        int[] transposedData = new int[data.length];
        int[] srcStrides = {shape[1] * shape[2], shape[2], 1};
        int[] dstStrides = {transposedShape[1] * transposedShape[2], transposedShape[2], 1};

        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                for (int k = 0; k < shape[2]; k++) {
                    int[] srcCoords = {i, j, k};
                    int[] dstCoords = new int[3];
                    dstCoords[0] = srcCoords[dstAxes[0]];
                    dstCoords[1] = srcCoords[dstAxes[1]];
                    dstCoords[2] = srcCoords[dstAxes[2]];

                    int srcIndex = i * srcStrides[0] + j * srcStrides[1] + k * srcStrides[2];
                    int dstIndex = dstCoords[0] * dstStrides[0] + dstCoords[1] * dstStrides[1] + dstCoords[2] * dstStrides[2];
                    
                    transposedData[dstIndex] = data[srcIndex];
                }
            }
        }
        return transposedData;
    }

    private float[] normalizeFloatArray(float[] array) {
        if (array == null || array.length == 0) {
            return array;
        }

        float maxVal = 0.0f;
        for (float v : array) {
            if (Math.abs(v) > maxVal) {
                maxVal = Math.abs(v);
            }
        }

        if (maxVal > 1.0f) {
            Log.d(TAG, "音频数据归一化，原始最大绝对值: " + maxVal);
            float[] normalizedArray = new float[array.length];
            for (int i = 0; i < array.length; i++) {
                normalizedArray[i] = array[i] / maxVal;
            }
            return normalizedArray;
        } else {
            Log.d(TAG, "音频数据无需归一化，最大绝对值: " + maxVal);
            return array;
        }
    }

    @Override
    public void onClick(View view) {
        if (view == mEncodeButton) {
            mResultText.setText("正在编码...");
            mTimeText.setText("");
            new AudioEncodeTask().execute();
        } else if (view == mImportWavButton) {
            importWavFile();
        } else if (view == mShareResultButton) {
            shareResultFile();
        } else if (view == mImportCodesButton) {
            importCodesFile();
        } else if (view == mDecodeButton) {
            mResultText.setText("正在解码...");
            mTimeText.setText("");
            new AudioDecodeTask().execute();
        } else if (view == mPlayAudioButton) {
            playDecodedAudio();
        } else if (view == mSaveAudioButton) {
            saveDecodedAudio();
        }
    }

    private void importWavFile() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("audio/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        try {
            startActivityForResult(Intent.createChooser(intent, "选择WAV音频文件"), REQUEST_IMPORT_WAV);
        } catch (android.content.ActivityNotFoundException ex) {
            Toast.makeText(this, "请安装文件管理器", Toast.LENGTH_SHORT).show();
        }
    }
    
    private void shareResultFile() {
        if (mLastResultFilePath == null || !new File(mLastResultFilePath).exists()) {
            Toast.makeText(this, "没有可分享的编码结果文件", Toast.LENGTH_SHORT).show();
            return;
        }
        
        try {
            File file = new File(mLastResultFilePath);
            Uri fileUri = FileProvider.getUriForFile(this, 
                getPackageName() + ".fileprovider", file);
                
            Intent shareIntent = new Intent(Intent.ACTION_SEND);
            shareIntent.setType("text/plain");
            shareIntent.putExtra(Intent.EXTRA_STREAM, fileUri);
            shareIntent.putExtra(Intent.EXTRA_SUBJECT, "音频编码结果");
            shareIntent.putExtra(Intent.EXTRA_TEXT, "这是使用FunCodec编码的音频结果文件。");
            shareIntent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            
            startActivity(Intent.createChooser(shareIntent, "分享编码结果"));
        } catch (Exception e) {
            Log.e(TAG, "分享文件失败", e);
            Toast.makeText(this, "分享失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        
        if (requestCode == REQUEST_IMPORT_WAV && resultCode == RESULT_OK) {
            if (data != null && data.getData() != null) {
                Uri uri = data.getData();
                importWavFromUri(uri);
            }
        } else if (requestCode == REQUEST_IMPORT_CODES && resultCode == RESULT_OK) {
            if (data != null && data.getData() != null) {
                Uri uri = data.getData();
                importCodesFromUri(uri);
            }
        }
    }
    
    private void importWavFromUri(Uri uri) {
        try {
            mStatusText.setText("正在导入音频文件...");
            
            // 复制文件到应用缓存目录
            InputStream inputStream = getContentResolver().openInputStream(uri);
            String fileName = "imported_audio_" + System.currentTimeMillis() + ".wav";
            File targetFile = new File(getCacheDir(), fileName);
            
            FileOutputStream outputStream = new FileOutputStream(targetFile);
            copyStream(inputStream, outputStream);
            inputStream.close();
            outputStream.close();
            
            // 验证并加载音频
            float[] audioData = AudioFileReader.loadWavFile(targetFile.getAbsolutePath(), 16000);
            if (audioData != null) {
                mAudioData = audioData;
                mWavPath = targetFile.getAbsolutePath();
                
                float duration = audioData.length / 16000.0f;
                mStatusText.setText(String.format("音频导入成功！时长: %.2f秒, 采样数: %d", 
                    duration, audioData.length));
                
                // 启用编码按钮
                mEncodeButton.setEnabled(true);
                
                Log.d(TAG, "导入音频成功: " + targetFile.getAbsolutePath() + 
                    ", 时长: " + duration + "秒");
                
            } else {
                mStatusText.setText("音频文件格式不支持或读取失败");
                Toast.makeText(this, "音频文件格式不支持", Toast.LENGTH_SHORT).show();
            }
            
        } catch (Exception e) {
            Log.e(TAG, "导入音频文件失败", e);
            mStatusText.setText("导入失败: " + e.getMessage());
            Toast.makeText(this, "导入失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private void importCodesFile() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("text/*");
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        try {
            startActivityForResult(Intent.createChooser(intent, "选择编码文件"), REQUEST_IMPORT_CODES);
        } catch (android.content.ActivityNotFoundException ex) {
            Toast.makeText(this, "请安装文件管理器", Toast.LENGTH_SHORT).show();
        }
    }
    
    private void importCodesFromUri(Uri uri) {
        try {
            mStatusText.setText("正在导入编码文件...");
            
            // 复制文件到应用缓存目录
            InputStream inputStream = getContentResolver().openInputStream(uri);
            String fileName = "imported_codes_" + System.currentTimeMillis() + ".txt";
            File targetFile = new File(getCacheDir(), fileName);
            
            FileOutputStream outputStream = new FileOutputStream(targetFile);
            copyStream(inputStream, outputStream);
            inputStream.close();
            outputStream.close();
            
            // 读取并解析编码数据
            List<Integer> codesList = new ArrayList<>();
            int[] importedShape = null;

            // 使用 BufferedReader 逐行读取，更健壮、内存效率更高
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(targetFile)))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (line.startsWith("shape:")) {
                        try {
                            String shapeStr = line.substring("shape:".length());
                            String[] dims = shapeStr.split(",");
                            if (dims.length == 3) {
                                importedShape = new int[3];
                                importedShape[0] = Integer.parseInt(dims[0].trim());
                                importedShape[1] = Integer.parseInt(dims[1].trim());
                                importedShape[2] = Integer.parseInt(dims[2].trim());
                            }
                        } catch (Exception e) {
                            Log.w(TAG, "解析shape失败: " + line, e);
                        }
                    } else if (!line.isEmpty() && !line.startsWith("#")) {
                        try {
                            codesList.add(Integer.parseInt(line));
                        } catch (NumberFormatException e) {
                            // 跳过无效行，并记录日志
                            Log.w(TAG, "解析编码行失败，跳过: '" + line + "'", e);
                        }
                    }
                }
            }
            
            if (!codesList.isEmpty() && importedShape != null) {
                mCodesData = new int[codesList.size()];
                for (int i = 0; i < codesList.size(); i++) {
                    mCodesData[i] = codesList.get(i);
                }
                mCodesShape = importedShape;
                mLastCodesFilePath = targetFile.getAbsolutePath();

                // 验证数据长度和形状
                int expectedLength = importedShape[0] * importedShape[1] * importedShape[2];
                if (mCodesData.length == expectedLength) {
                    int quantizerLayers = importedShape[0];
                    int totalFrames = importedShape[2];
                    mStatusText.setText(String.format("编码导入成功！编码数: %d, 帧数: %d, 层数: %d",
                            mCodesData.length, totalFrames, quantizerLayers));

                    // 启用解码按钮
                    mDecodeButton.setEnabled(mDecoderSession != null);

                    Log.d(TAG, "导入编码成功: " + targetFile.getAbsolutePath() +
                            ", 编码数量: " + mCodesData.length + ", 形状: " + java.util.Arrays.toString(mCodesShape));
                } else {
                    mStatusText.setText(String.format("编码数据格式不正确 (数量 %d 与形状 %s 不匹配)",
                            mCodesData.length, java.util.Arrays.toString(importedShape)));
                    mDecodeButton.setEnabled(false);
                }
            } else {
                mStatusText.setText("编码文件中没有找到有效数据或形状信息");
                Toast.makeText(this, "编码文件格式不正确", Toast.LENGTH_SHORT).show();
            }
            
        } catch (Exception e) {
            Log.e(TAG, "导入编码文件失败", e);
            mStatusText.setText("导入失败: " + e.getMessage());
            Toast.makeText(this, "导入失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }
    
    private void playDecodedAudio() {
        if (mDecodedAudioData == null || mDecodedAudioData.length == 0) {
            Toast.makeText(this, "没有可播放的音频数据", Toast.LENGTH_SHORT).show();
            return;
        }
        
        if (mAudioPlayer.isPlaying()) {
            mAudioPlayer.stopPlayback();
            mPlayAudioButton.setText("播放音频");
        } else {
            mAudioPlayer.playAudioData(mDecodedAudioData);
            mPlayAudioButton.setText("播放音频");
            
            // 5秒后自动恢复按钮文本（音频播放完成后）
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        Thread.sleep(5000);
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                if (!mAudioPlayer.isPlaying()) {
                                    mPlayAudioButton.setText("播放音频");
                                }
                            }
                        });
                    } catch (InterruptedException e) {
                        // 忽略
                    }
                }
            }).start();
        }
    }
    
    private void saveDecodedAudio() {
        if (mDecodedAudioData == null || mDecodedAudioData.length == 0) {
            Toast.makeText(this, "没有可保存的音频数据", Toast.LENGTH_SHORT).show();
            return;
        }
        
        try {
            // 使用应用私有目录
            File outputDir = new File(getExternalFilesDir(null), "decoded_audio");
            if (!outputDir.exists()) {
                outputDir.mkdirs();
            }
            
            // 生成带时间戳的文件名
            String timestamp = String.valueOf(System.currentTimeMillis());
            String fileName = "decoded_audio_" + timestamp + ".wav";
            File outputFile = new File(outputDir, fileName);
            
            boolean success = AudioPlayer.saveAsWav(mDecodedAudioData, outputFile.getAbsolutePath());
            
            if (success) {
                Toast.makeText(this, "音频已保存到应用目录", Toast.LENGTH_SHORT).show();
                Log.d(TAG, "解码音频已保存到: " + outputFile.getAbsolutePath());
            } else {
                Toast.makeText(this, "音频保存失败", Toast.LENGTH_SHORT).show();
            }
            
        } catch (Exception e) {
            Log.e(TAG, "保存解码音频失败", e);
            Toast.makeText(this, "保存失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        
        // 停止音频播放
        if (mAudioPlayer != null) {
            mAudioPlayer.stopPlayback();
        }
        
        // 释放编码器资源
        if (mNetInstance != null) {
            mNetInstance.release();
            mNetInstance = null;
        }
        
        // 释放解码器资源
        if (mDecoderNetInstance != null) {
            mDecoderNetInstance.release();
            mDecoderNetInstance = null;
        }
    }

    /**
     * 复制大型assets文件到缓存目录
     * 专门用于处理100MB+的模型文件
     */
    private boolean copyLargeAssetFile(String assetFileName, String targetPath) {
        InputStream inputStream = null;
        FileOutputStream outputStream = null;
        
        try {
            Log.d(TAG, "开始复制大文件: " + assetFileName + " -> " + targetPath);
            
            // 打开assets文件
            inputStream = getAssets().open(assetFileName);
            
            // 创建输出文件
            File targetFile = new File(targetPath);
            File parentDir = targetFile.getParentFile();
            if (parentDir != null && !parentDir.exists()) {
                parentDir.mkdirs();
            }
            
            outputStream = new FileOutputStream(targetFile);
            
            // 使用较大的缓冲区复制文件
            byte[] buffer = new byte[64 * 1024]; // 64KB缓冲区
            int bytesRead;
            long totalBytesRead = 0;
            long lastLogTime = System.currentTimeMillis();
            
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
                totalBytesRead += bytesRead;
                
                // 每5秒输出一次进度
                long currentTime = System.currentTimeMillis();
                if (currentTime - lastLogTime > 5000) {
                    Log.d(TAG, "复制进度: " + (totalBytesRead / 1024 / 1024) + " MB");
                    lastLogTime = currentTime;
                }
            }
            
            outputStream.flush();
            
            // 验证复制结果
            if (targetFile.exists() && targetFile.length() > 100000000) { // 至少100MB
                Log.d(TAG, "大文件复制成功: " + targetPath + ", 大小: " + targetFile.length() + " 字节");
                return true;
            } else {
                Log.e(TAG, "大文件复制验证失败: 目标文件不存在或大小不正确");
                return false;
            }
            
        } catch (Exception e) {
            Log.e(TAG, "大文件复制失败: " + assetFileName, e);
            return false;
        } finally {
            try {
                if (inputStream != null) {
                    inputStream.close();
                }
                if (outputStream != null) {
                    outputStream.close();
                }
            } catch (IOException e) {
                Log.e(TAG, "关闭文件流失败", e);
            }
        }
    }
} 