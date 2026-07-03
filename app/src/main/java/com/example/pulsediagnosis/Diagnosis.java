package com.example.pulsediagnosis;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class Diagnosis extends AppCompatActivity {

    TextView tvPrediction;
    TextView tvStatus;
    TextView tvRisk;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_diagnosis);

        tvPrediction = findViewById(R.id.tvPrediction);
        tvStatus = findViewById(R.id.tvStatus);
        tvRisk = findViewById(R.id.tvRisk);

        loadDiagnosis();
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadDiagnosis(); // ensures updated data when returning to this screen
    }

    private void loadDiagnosis() {

        SharedPreferences prefs =
                getSharedPreferences("DiagnosisData", MODE_PRIVATE);

        if (!prefs.contains("prediction")) {
            showEmptyState();
            return;
        }
        String prediction = prefs.getString("prediction", "Unknown");
        String status = prefs.getString("status", "Unknown");
        String risk = prefs.getString("risk", "Unknown");

        if ("Unknown".equals(prediction)) {
            showEmptyState();
            return;
        }

        tvPrediction.setText("Prediction: " + prediction);
        tvStatus.setText("Status: " + status);
        tvRisk.setText("Risk Level: " + risk);
    }

    private void showEmptyState() {
        tvPrediction.setText("No Diagnosis Available");
        tvStatus.setText("");
        tvRisk.setText("");
    }
}