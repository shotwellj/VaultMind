/**
 * Capture Screen - Photo-to-Knowledge and Call Recording
 *
 * Snap documents, whiteboards, business cards, or record calls.
 * Everything processes locally, only extracted text syncs to your server.
 */

import React, { useState } from "react";
import {
  View, Text, TouchableOpacity, StyleSheet, ScrollView, Alert,
} from "react-native";
import { uploadPhoto, processTranscript } from "../services/api";
import { queueItem } from "../services/sync";

export default function CaptureScreen() {
  const [processing, setProcessing] = useState(false);
  const [lastResult, setLastResult] = useState(null);

  const handlePhotoCapture = async () => {
    // In production, this would use expo-camera
    // For now, show the flow
    Alert.alert(
      "Capture Document",
      "Point your camera at a document, whiteboard, business card, or receipt.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Simulate Capture",
          onPress: async () => {
            setProcessing(true);
            try {
              // This would send actual camera data in production
              const result = await uploadPhoto(
                null, // base64 camera data
                `capture-${Date.now()}.jpg`,
                "auto"
              );
              setLastResult(result);
            } catch (e) {
              // Queue for later sync if offline
              await queueItem({
                type: "photo",
                filename: `capture-${Date.now()}.jpg`,
                queued_reason: e.message,
              });
              Alert.alert("Queued", "Photo queued for processing when you reconnect.");
            } finally {
              setProcessing(false);
            }
          },
        },
      ]
    );
  };

  const handleVoiceNote = async () => {
    Alert.alert(
      "Record Voice Note",
      "Record a voice memo or call summary. It will be transcribed and added to your vault.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Simulate Recording",
          onPress: async () => {
            setProcessing(true);
            try {
              const result = await processTranscript(
                "This is a sample transcript from a voice recording.",
                { type: "voice_note", date: new Date().toISOString() }
              );
              setLastResult(result);
            } catch (e) {
              await queueItem({
                type: "voice_note",
                queued_reason: e.message,
              });
              Alert.alert("Queued", "Recording queued for processing when you reconnect.");
            } finally {
              setProcessing(false);
            }
          },
        },
      ]
    );
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Capture Knowledge</Text>
      <Text style={styles.subtitle}>
        Turn physical documents and conversations into searchable intelligence.
      </Text>

      <TouchableOpacity
        style={[styles.captureBtn, styles.photoBtn]}
        onPress={handlePhotoCapture}
        disabled={processing}
      >
        <Text style={styles.btnEmoji}>📸</Text>
        <View style={styles.btnTextWrap}>
          <Text style={styles.btnTitle}>Scan Document</Text>
          <Text style={styles.btnDesc}>
            Contracts, whiteboards, business cards, receipts
          </Text>
        </View>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.captureBtn, styles.voiceBtn]}
        onPress={handleVoiceNote}
        disabled={processing}
      >
        <Text style={styles.btnEmoji}>🎙️</Text>
        <View style={styles.btnTextWrap}>
          <Text style={styles.btnTitle}>Voice Note</Text>
          <Text style={styles.btnDesc}>
            Record a memo or call summary
          </Text>
        </View>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.captureBtn, styles.callBtn]}
        disabled={processing}
      >
        <Text style={styles.btnEmoji}>📞</Text>
        <View style={styles.btnTextWrap}>
          <Text style={styles.btnTitle}>Call Transcript</Text>
          <Text style={styles.btnDesc}>
            Paste or import a call transcript
          </Text>
        </View>
      </TouchableOpacity>

      {processing && (
        <View style={styles.processingBanner}>
          <Text style={styles.processingText}>Processing locally...</Text>
        </View>
      )}

      {lastResult && (
        <View style={styles.resultCard}>
          <Text style={styles.resultTitle}>Last Capture</Text>
          <Text style={styles.resultText}>
            {lastResult.document_type || "document"} processed
          </Text>
          {lastResult.text && (
            <Text style={styles.resultPreview} numberOfLines={4}>
              {lastResult.text}
            </Text>
          )}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0e17" },
  content: { padding: 20 },
  title: { color: "#f9fafb", fontSize: 24, fontWeight: "700", marginBottom: 4 },
  subtitle: { color: "#9ca3af", fontSize: 14, marginBottom: 24, lineHeight: 20 },
  captureBtn: {
    flexDirection: "row", alignItems: "center", padding: 18,
    borderRadius: 12, marginBottom: 12, borderWidth: 1,
  },
  photoBtn: { backgroundColor: "#1e3a5f20", borderColor: "#1e3a5f" },
  voiceBtn: { backgroundColor: "#7c3aed20", borderColor: "#7c3aed" },
  callBtn: { backgroundColor: "#059669 20", borderColor: "#059669" },
  btnEmoji: { fontSize: 28, marginRight: 14 },
  btnTextWrap: { flex: 1 },
  btnTitle: { color: "#f9fafb", fontSize: 17, fontWeight: "600" },
  btnDesc: { color: "#9ca3af", fontSize: 13, marginTop: 2 },
  processingBanner: {
    backgroundColor: "#1f2937", padding: 12, borderRadius: 8,
    marginTop: 12, alignItems: "center",
  },
  processingText: { color: "#60a5fa", fontSize: 14 },
  resultCard: {
    backgroundColor: "#111827", padding: 16, borderRadius: 12,
    marginTop: 20, borderWidth: 1, borderColor: "#1f2937",
  },
  resultTitle: { color: "#f9fafb", fontSize: 16, fontWeight: "600", marginBottom: 6 },
  resultText: { color: "#9ca3af", fontSize: 14 },
  resultPreview: { color: "#6b7280", fontSize: 12, marginTop: 8, fontStyle: "italic" },
});
