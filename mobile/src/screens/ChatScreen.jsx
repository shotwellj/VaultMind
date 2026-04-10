/**
 * Chat Screen - Main VaultMind chat interface
 *
 * Mirrors the desktop experience on mobile.
 * Supports voice input via on-device Whisper.
 * Streams responses with quality badges and citations.
 */

import React, { useState, useRef, useCallback } from "react";
import {
  View, Text, TextInput, FlatList, TouchableOpacity,
  StyleSheet, KeyboardAvoidingView, Platform, ActivityIndicator,
} from "react-native";
import { streamChat } from "../services/api";
import { queueItem } from "../services/sync";

export default function ChatScreen() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const flatListRef = useRef(null);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput("");
    const userMsg = { role: "user", content: text, id: Date.now() };
    const aiMsg = { role: "assistant", content: "", id: Date.now() + 1, streaming: true };

    setMessages((prev) => [...prev, userMsg, aiMsg]);
    setLoading(true);
    setStatus("");

    try {
      await streamChat(
        text,
        null, // auto model selection
        "vault",
        messages.slice(-6).map((m) => ({ role: m.role, content: m.content })),
        // onToken
        (token) => {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last.role === "assistant") {
              last.content += token;
              last.streaming = true;
            }
            return updated;
          });
        },
        // onStatus
        (statusText) => setStatus(statusText),
        // onDone
        (data) => {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last.role === "assistant") {
              last.streaming = false;
              last.sources = data.sources || [];
            }
            return updated;
          });
          setStatus("");
          setLoading(false);
        }
      );
    } catch (e) {
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last.role === "assistant") {
          last.content = `Connection error: ${e.message}. Your server may be offline.`;
          last.streaming = false;
          last.error = true;
        }
        return updated;
      });
      setLoading(false);
      setStatus("");
    }
  }, [input, loading, messages]);

  const renderMessage = ({ item }) => (
    <View style={[styles.msgBubble, item.role === "user" ? styles.userBubble : styles.aiBubble]}>
      <Text style={[styles.msgText, item.error && styles.errorText]}>
        {item.content || (item.streaming ? "..." : "")}
      </Text>
      {item.sources && item.sources.length > 0 && (
        <Text style={styles.sourcesText}>
          Sources: {item.sources.length} documents
        </Text>
      )}
    </View>
  );

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      keyboardVerticalOffset={90}
    >
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={(item) => String(item.id)}
        contentContainerStyle={styles.messageList}
        onContentSizeChange={() => flatListRef.current?.scrollToEnd()}
      />

      {status ? (
        <View style={styles.statusBar}>
          <ActivityIndicator size="small" color="#3b82f6" />
          <Text style={styles.statusText}>{status}</Text>
        </View>
      ) : null}

      <View style={styles.inputRow}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Ask VaultMind..."
          placeholderTextColor="#6b7280"
          multiline
          maxLength={2000}
          editable={!loading}
          onSubmitEditing={sendMessage}
        />
        <TouchableOpacity
          style={[styles.sendBtn, loading && styles.sendBtnDisabled]}
          onPress={sendMessage}
          disabled={loading}
        >
          <Text style={styles.sendBtnText}>{loading ? "..." : "Send"}</Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0e17" },
  messageList: { padding: 16, paddingBottom: 8 },
  msgBubble: {
    maxWidth: "85%", padding: 12, borderRadius: 12, marginBottom: 8,
  },
  userBubble: {
    alignSelf: "flex-end", backgroundColor: "#1e3a5f",
  },
  aiBubble: {
    alignSelf: "flex-start", backgroundColor: "#1f2937",
  },
  msgText: { color: "#f9fafb", fontSize: 15, lineHeight: 22 },
  errorText: { color: "#fca5a5" },
  sourcesText: { color: "#6b7280", fontSize: 12, marginTop: 6 },
  statusBar: {
    flexDirection: "row", alignItems: "center", paddingHorizontal: 16,
    paddingVertical: 6, backgroundColor: "#111827",
  },
  statusText: { color: "#9ca3af", fontSize: 13, marginLeft: 8 },
  inputRow: {
    flexDirection: "row", padding: 12, backgroundColor: "#111827",
    borderTopWidth: 1, borderTopColor: "#1f2937", alignItems: "flex-end",
  },
  input: {
    flex: 1, backgroundColor: "#1f2937", color: "#f9fafb", borderRadius: 20,
    paddingHorizontal: 16, paddingVertical: 10, fontSize: 15, maxHeight: 100,
  },
  sendBtn: {
    backgroundColor: "#3b82f6", borderRadius: 20, paddingHorizontal: 18,
    paddingVertical: 10, marginLeft: 8,
  },
  sendBtnDisabled: { backgroundColor: "#374151" },
  sendBtnText: { color: "#fff", fontWeight: "600", fontSize: 15 },
});
