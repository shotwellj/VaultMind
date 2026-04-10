/**
 * VaultMind Mobile App
 *
 * Your private AI in your pocket, synced to your vault.
 * No Apple cloud. No Google cloud. Just your phone talking to your server.
 */

import React, { useEffect, useState } from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { StatusBar } from "react-native";

import ChatScreen from "./screens/ChatScreen";
import VaultScreen from "./screens/VaultScreen";
import CaptureScreen from "./screens/CaptureScreen";
import AlertsScreen from "./screens/AlertsScreen";
import SettingsScreen from "./screens/SettingsScreen";
import { initSync, runSyncCycle } from "./services/sync";
import { getUnreadCount } from "./services/api";

const Tab = createBottomTabNavigator();

const DARK_THEME = {
  dark: true,
  colors: {
    primary: "#3b82f6",
    background: "#0a0e17",
    card: "#111827",
    text: "#f9fafb",
    border: "#1f2937",
    notification: "#ef4444",
  },
};

export default function App() {
  const [unreadAlerts, setUnreadAlerts] = useState(0);

  useEffect(() => {
    // Initialize sync on app start
    initSync().catch(console.warn);

    // Run sync cycle every 5 minutes
    const syncInterval = setInterval(() => {
      runSyncCycle().catch(console.warn);
    }, 5 * 60 * 1000);

    // Check for unread alerts every 60 seconds
    const alertInterval = setInterval(async () => {
      try {
        const result = await getUnreadCount();
        setUnreadAlerts(result.unread || 0);
      } catch {}
    }, 60 * 1000);

    return () => {
      clearInterval(syncInterval);
      clearInterval(alertInterval);
    };
  }, []);

  return (
    <>
      <StatusBar barStyle="light-content" backgroundColor="#0a0e17" />
      <NavigationContainer theme={DARK_THEME}>
        <Tab.Navigator
          screenOptions={{
            headerStyle: { backgroundColor: "#111827", borderBottomColor: "#1f2937" },
            headerTintColor: "#f9fafb",
            tabBarStyle: { backgroundColor: "#111827", borderTopColor: "#1f2937" },
            tabBarActiveTintColor: "#3b82f6",
            tabBarInactiveTintColor: "#6b7280",
          }}
        >
          <Tab.Screen
            name="Chat"
            component={ChatScreen}
            options={{ tabBarLabel: "Chat", title: "VaultMind" }}
          />
          <Tab.Screen
            name="Vault"
            component={VaultScreen}
            options={{ tabBarLabel: "Vault", title: "Your Vault" }}
          />
          <Tab.Screen
            name="Capture"
            component={CaptureScreen}
            options={{ tabBarLabel: "Capture", title: "Capture" }}
          />
          <Tab.Screen
            name="Alerts"
            component={AlertsScreen}
            options={{
              tabBarLabel: "Alerts",
              title: "Alerts",
              tabBarBadge: unreadAlerts > 0 ? unreadAlerts : undefined,
            }}
          />
          <Tab.Screen
            name="Settings"
            component={SettingsScreen}
            options={{ tabBarLabel: "Settings", title: "Settings" }}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </>
  );
}
