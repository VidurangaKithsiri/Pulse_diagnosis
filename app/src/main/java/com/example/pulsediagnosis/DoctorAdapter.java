package com.example.pulsediagnosis;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;


public class DoctorAdapter extends RecyclerView.Adapter<DoctorAdapter.ViewHolder> {

    ArrayList<Doctor> doctorList;
    Context context;

    public DoctorAdapter(Context context,ArrayList<Doctor> doctorList) {
        this.doctorList = doctorList;
        this.context = context;

    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_doctor, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        Doctor doctor = doctorList.get(position);
        holder.tvName.setText(doctor.name);
        holder.tvEmail.setText(doctor.email);
        holder.tvContact.setText(doctor.contact); // change if contact field exists// change if contact field exists
        holder.tvAbout.setText(doctor.about);


        holder.itemView.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_DIAL);
            intent.setData(Uri.parse("tel:" + doctor.about));
            context.startActivity(intent);
        });    }

    @Override
    public int getItemCount() {
        return doctorList.size();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {

        TextView tvName, tvEmail, tvAbout, tvContact;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            tvName = itemView.findViewById(R.id.tvName);
            tvEmail = itemView.findViewById(R.id.tvEmail);
            tvAbout = itemView.findViewById(R.id.tvAbout);
            tvContact = itemView.findViewById(R.id.tvContact);
        }
    }
}
