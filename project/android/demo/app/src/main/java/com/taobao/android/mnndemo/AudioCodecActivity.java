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

import com.taobao.android.mnn.MNNForwardType;
import com.taobao.android.mnn.MNNNetInstance;
import com.taobao.android.utils.AudioFileReader;
import com.taobao.android.utils.AudioPlayer;
import com.taobao.android.utils.Common;
import com.taobao.android.utils.PermissionUtils;
import com.taobao.android.utils.WavFileGenerator;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
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
    private float[] mDecodedAudioData; // 解码后的音频数据
    private String mLastResultFilePath; // 最近的编码结果文件路径
    private String mLastCodesFilePath; // 最近导入的编码文件路径
    private AudioPlayer mAudioPlayer;

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
                  .append(", ").append(String.format("%.3f", getArrayMaxFloat(result.waveform))).append("]\n\n");
                
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
        
        mEncodeButton.setOnClickListener(this);
        mImportWavButton.setOnClickListener(this);
        mShareResultButton.setOnClickListener(this);
        mImportCodesButton.setOnClickListener(this);
        mDecodeButton.setOnClickListener(this);
        mPlayAudioButton.setOnClickListener(this);
        mSaveAudioButton.setOnClickListener(this);
        
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
                    Common.copyAssetResource2File(this, MODEL_FILE_NAME, mModelPath);
                    
                    // 验证复制结果
                    if (cacheModel.exists() && cacheModel.length() > 100000000) {
                        Log.d(TAG, "编码器从assets复制成功, 大小: " + cacheModel.length());
                        assetsSuccess = true;
                    }
                } catch (Exception e) {
                    Log.w(TAG, "编码器从assets复制失败: " + e.getMessage());
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
                    Common.copyAssetResource2File(this, DECODER_MODEL_FILE_NAME, mDecoderModelPath);
                    
                    // 验证复制结果
                    if (cacheDecoderModel.exists() && cacheDecoderModel.length() > 1000000) {
                        Log.d(TAG, "解码器从assets复制成功, 大小: " + cacheDecoderModel.length());
                        decoderAssetsSuccess = true;
                    }
                } catch (Exception e) {
                    Log.w(TAG, "解码器从assets复制失败: " + e.getMessage());
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

            // 创建编码器Session
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

            // 创建解码器MNN实例（如果解码器模型存在）
            File finalDecoderModelFile = new File(mDecoderModelPath);
            if (finalDecoderModelFile.exists() && finalDecoderModelFile.length() > 1000000) { // 至少1MB
                Log.d(TAG, "开始创建解码器MNN实例...");
                Log.d(TAG, "解码器模型文件: " + mDecoderModelPath + ", 大小: " + finalDecoderModelFile.length() + " 字节");
                
                try {
                    mDecoderNetInstance = MNNNetInstance.createFromFile(mDecoderModelPath);
                    if (mDecoderNetInstance != null) {
                        Log.d(TAG, "解码器MNN实例创建成功");
                        
                        mDecoderSession = mDecoderNetInstance.createSession(config);
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
            
            // 检查输出tensor是否也正确初始化了
            int[] outputShape = mOutputTensor.getDimensions();
            Log.d(TAG, "输出tensor形状: " + java.util.Arrays.toString(outputShape));
            
            // 设置输入数据 - 使用更安全的方式
            Log.d(TAG, "设置输入数据...");
            float[] inputData = new float[mAudioData.length];
            System.arraycopy(mAudioData, 0, inputData, 0, mAudioData.length);
            
            // 尝试使用writeBufferToTensor而不是setInputFloatData
            try {
                mInputTensor.setInputFloatData(inputData);
                Log.d(TAG, "输入数据设置成功");
            } catch (Exception e) {
                Log.e(TAG, "setInputFloatData失败，尝试其他方法: " + e.getMessage());
                // 如果setInputFloatData失败，可能需要使用其他方法
                throw e;
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
            
        } catch (Exception e) {
            Log.e(TAG, "音频编码异常", e);
            result.errorMessage = "编码异常: " + e.getMessage();
            e.printStackTrace();
        }
        
        return result;
    }

    private AudioDecodeResult performAudioDecode() {
        AudioDecodeResult result = new AudioDecodeResult();
        
        try {
            if (mDecoderSession == null || mCodesData == null) {
                result.errorMessage = "解码器未准备好或编码数据为空";
                return result;
            }
            
            Log.d(TAG, "开始音频解码...");
            Log.d(TAG, "编码数据长度: " + mCodesData.length);
            
            // 根据编码数据重塑为 [1, 32, frames] 形状
            int totalFrames = mCodesData.length / 32;
            if (mCodesData.length % 32 != 0) {
                result.errorMessage = "编码数据长度不是32的倍数，无法重塑为正确形状";
                return result;
            }
            
            int[] inputShape = {1, 32, totalFrames};
            Log.d(TAG, "调整解码器输入tensor形状为: " + java.util.Arrays.toString(inputShape));
            
            mDecoderInputTensor.reshape(inputShape);
            mDecoderSession.reshape();
            
            // 设置输入数据
            Log.d(TAG, "设置编码数据...");
            mDecoderInputTensor.setInputIntData(mCodesData);
            
            // 执行推理
            Log.d(TAG, "开始解码推理...");
            long startTime = System.nanoTime();
            mDecoderSession.run();
            long endTime = System.nanoTime();
            
            result.inferenceTime = (endTime - startTime) / 1000000.0f;
            
            // 获取输出数据
            Log.d(TAG, "获取解码结果...");
            result.waveform = mDecoderOutputTensor.getFloatData();
            result.outputShape = mDecoderOutputTensor.getDimensions();
            
            // 计算音频时长和实时率
            result.audioDuration = result.waveform.length / 16000.0f; // 16kHz采样率
            result.realTimeRatio = result.audioDuration * 1000 / result.inferenceTime;
            
            Log.d(TAG, "解码完成，耗时: " + result.inferenceTime + "ms");
            Log.d(TAG, "音频时长: " + result.audioDuration + "秒");
            Log.d(TAG, "实时率: " + result.realTimeRatio + "x");
            Log.d(TAG, "输出形状: " + java.util.Arrays.toString(result.outputShape));
            Log.d(TAG, "输出数据长度: " + result.waveform.length);
            
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
            try (FileInputStream fis = new FileInputStream(targetFile)) {
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = fis.read(buffer)) != -1) {
                    baos.write(buffer, 0, bytesRead);
                }
                
                String content = baos.toString();
                String[] lines = content.split("\n");
                
                for (String line : lines) {
                    line = line.trim();
                    if (!line.isEmpty() && !line.startsWith("#")) {
                        try {
                            codesList.add(Integer.parseInt(line));
                        } catch (NumberFormatException e) {
                            // 跳过无效行
                        }
                    }
                }
            }
            
            if (codesList.size() > 0) {
                mCodesData = new int[codesList.size()];
                for (int i = 0; i < codesList.size(); i++) {
                    mCodesData[i] = codesList.get(i);
                }
                
                mLastCodesFilePath = targetFile.getAbsolutePath();
                
                // 检查数据格式
                if (mCodesData.length % 32 == 0) {
                    int totalFrames = mCodesData.length / 32;
                    mStatusText.setText(String.format("编码文件导入成功！编码数量: %d, 帧数: %d", 
                        mCodesData.length, totalFrames));
                    
                    // 启用解码按钮
                    mDecodeButton.setEnabled(mDecoderSession != null);
                    
                    Log.d(TAG, "导入编码成功: " + targetFile.getAbsolutePath() + 
                        ", 编码数量: " + mCodesData.length);
                        
                } else {
                    mStatusText.setText("编码数据格式不正确（长度不是32的倍数）");
                    mDecodeButton.setEnabled(false);
                }
                
            } else {
                mStatusText.setText("编码文件中没有找到有效数据");
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
            mPlayAudioButton.setText("停止播放");
            
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
} 