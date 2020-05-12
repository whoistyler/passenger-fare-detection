package com.example.nfc_payment;

import android.os.AsyncTask;
import android.os.Handler;
import android.util.Log;
import android.widget.Toast;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.Socket;

public class MessageSender extends AsyncTask<String, Void, Void> {
    private Exception exception;
    @Override
    protected Void doInBackground(String... params) {
        try {
            try {
                // Add your own socket id
                Socket socket = new Socket("000.000.0.00", 8888);
                PrintWriter outToServer = new PrintWriter(
                        new OutputStreamWriter(
                                socket.getOutputStream()));
                outToServer.print(params[0]);
                outToServer.flush();
                Log.d("SENT MSG: ", params[0]);

            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (Exception e) {
            this.exception = e;
            return null;
        }
        return null;
    }
}
