package com.example.imagerecog;

import static org.tensorflow.lite.DataType.UINT8;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.renderscript.Element;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.imagerecog.ml.MobilenetV110224Quant;
import com.google.android.gms.fitness.data.DataType;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    private ImageView imgView;
    private Button select, predict;
    private TextView tv;
    private Bitmap img;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        String[] labels=new String[1001];
        int cnt=0;
        BufferedReader bufferedReader;


            try {
                bufferedReader = new BufferedReader(new InputStreamReader(getAssets().open("labels_mobilenet_quant_v1_224.txt")));
                String line = bufferedReader.readLine();
                while (line!=null) {
                    labels[cnt] = line;
                    cnt++;
                    line = bufferedReader.readLine();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

        setContentView(R.layout.activity_main);

        imgView = (ImageView) findViewById(R.id.imageView);
        tv = (TextView) findViewById(R.id.textView2);
        select = (Button) findViewById((R.id.button));
        predict = (Button) findViewById(R.id.button2);


        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 100);
            }





        });

        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                img= Bitmap.createScaledBitmap(img, 224,224,true);
                try {
                    MobilenetV110224Quant model = MobilenetV110224Quant.newInstance(getApplicationContext());

                    // Creates inputs for reference.

                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3},UINT8);

                    TensorImage tensorImage = new TensorImage(UINT8);
                    tensorImage.load(img);
                    ByteBuffer byteBuffer = tensorImage.getBuffer();




                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    MobilenetV110224Quant.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    // Releases model resources if no longer used.

                    tv.setText(labels[getmax(outputFeature0.getFloatArray())]+" ");
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }

            }
        });


    }
int getmax(float[] arr)
{
    int max=0;
    for(int i=0;i<arr.length;i++)
    {
        if(arr[i]>arr[max])
        {
            max=i;
        }
    }
    return max;
}
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode==100)
        {
            imgView.setImageURI(data.getData());
            Uri uri =data.getData();
            try {
                img = MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}

