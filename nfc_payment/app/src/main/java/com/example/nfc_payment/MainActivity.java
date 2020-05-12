package com.example.nfc_payment;

import android.media.AudioManager;
import android.media.ToneGenerator;
import android.os.Bundle;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.app.PendingIntent;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.Manifest;
import android.telephony.TelephonyManager;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import android.nfc.NfcAdapter;

public class MainActivity extends AppCompatActivity {

    private NfcAdapter mNfcAdapter;
    private PendingIntent pendingIntent;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button button = findViewById(R.id.button);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                triggerPayment();
            }
        });

        // Set the nfcAdapter and ensure the device supports it
        mNfcAdapter = NfcAdapter.getDefaultAdapter(this);
        if (mNfcAdapter == null) {
            //this device doesn't support NFC
            Toast.makeText(this,
                    "This device doesn't support NFC.",
                    Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        pendingIntent = PendingIntent.getActivity(
                this,
                0,
                new Intent(
                        this,
                        this.getClass()).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP), 0);
    }

    @Override
    protected void onResume(){
        super.onResume();

        if (mNfcAdapter != null){
            if (!mNfcAdapter.isEnabled()){
                //enable NFC??
            }
            mNfcAdapter.enableForegroundDispatch(
                    this, pendingIntent, null, null);
        }
    }

    @Override
    protected void onPause(){
        super.onPause();
        mNfcAdapter.disableForegroundDispatch(this);
    }

    @Override
    protected void onNewIntent(Intent intent){
        setIntent(intent);
        resolveIntent(intent);
    }

    private void resolveIntent(Intent intent){
        String action = intent.getAction();
        String imei = getImei();

        if (    mNfcAdapter.ACTION_TAG_DISCOVERED.equals(action) ||
                mNfcAdapter.ACTION_TECH_DISCOVERED.equals(action) ||
                mNfcAdapter.ACTION_NDEF_DISCOVERED.equals(action)){

            Toast.makeText(this, "Tap received", Toast.LENGTH_SHORT);
            triggerPayment();
        }
    }

    private void triggerPayment() {
        new MessageSender().execute(getImei());
        //Toast.makeText(this, "TYLER", Toast.LENGTH_SHORT);

        ToneGenerator toneGen1 = new ToneGenerator(AudioManager.STREAM_MUSIC, 100);
        toneGen1.startTone(ToneGenerator.TONE_CDMA_PIP,150);

        Intent myIntent = new Intent(this, paid_screen.class);
        myIntent.addFlags(Intent.FLAG_ACTIVITY_NO_ANIMATION);
        startActivity(myIntent);
    }

    private String getImei() {
        TelephonyManager telephonyManager;
        telephonyManager = (TelephonyManager) this.getSystemService(TELEPHONY_SERVICE);

        String imei = null;
        if (telephonyManager != null) {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.READ_PHONE_STATE) != PackageManager.PERMISSION_GRANTED) {

                // We do not have this permission. Let's ask the user
                ActivityCompat.requestPermissions(
                        this,
                        new String[]{Manifest.permission.READ_PHONE_STATE},
                        0);
            } else {
                imei = telephonyManager.getDeviceId();
            }
        }
        return "0";
        //return imei;
    }
}
