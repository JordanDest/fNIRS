#include "ESP32_NOW.h"
#include "WiFi.h"

#include <esp_mac.h>  // For the MAC2STR and MACSTR macros

/* Definitions */
#define ESPNOW_WIFI_CHANNEL 6
#define SENSOR_PIN1 34  // GPIO pin connected to the first set of sensors
#define SENSOR_PIN2 35  // GPIO pin connected to the second set of sensors
#define NUM_SENSORS 15  // Number of sensors in each set

/* Classes */

// Creating a new class that inherits from the ESP_NOW_Peer class is required.
class ESP_NOW_Broadcast_Peer : public ESP_NOW_Peer {
public:
  // Constructor of the class using the broadcast address
  ESP_NOW_Broadcast_Peer(uint8_t channel, wifi_interface_t iface, const uint8_t *lmk) : ESP_NOW_Peer(ESP_NOW.BROADCAST_ADDR, channel, iface, lmk) {}

  // Destructor of the class
  ~ESP_NOW_Broadcast_Peer() {
    remove();
  }

  // Function to properly initialize the ESP-NOW and register the broadcast peer
  bool begin() {
    if (!ESP_NOW.begin() || !add()) {
      log_e("Failed to initialize ESP-NOW or register the broadcast peer");
      return false;
    }
    return true;
  }

  // Function to send a message to all devices within the network
  bool send_message(const uint8_t *data, size_t len) {
    if (!send(data, len)) {
      log_e("Failed to broadcast message");
      return false;
    }
    return true;
  }
};

/* Global Variables */
uint32_t msg_count = 0;

// Create a broadcast peer object
ESP_NOW_Broadcast_Peer broadcast_peer(ESPNOW_WIFI_CHANNEL, WIFI_IF_STA, NULL);

/* Main */

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  // Initialize the Wi-Fi module
  WiFi.mode(WIFI_STA);
  WiFi.setChannel(ESPNOW_WIFI_CHANNEL);
  while (!WiFi.STA.started()) {
    delay(100);
  }

  // Set GPIO pins as input
  pinMode(SENSOR_PIN1, INPUT);
  pinMode(SENSOR_PIN2, INPUT);

  Serial.println("ESP-NOW Example - Broadcast Master");
  Serial.println("Wi-Fi parameters:");
  Serial.println("  Mode: STA");
  Serial.println("  MAC Address: " + WiFi.macAddress());
  Serial.printf("  Channel: %d\n", ESPNOW_WIFI_CHANNEL);

  // Register the broadcast peer
  if (!broadcast_peer.begin()) {
    Serial.println("Failed to initialize broadcast peer");
    Serial.println("Rebooting in 5 seconds...");
    delay(5000);
    ESP.restart();
  }

  Serial.println("Setup complete. Broadcasting sensor data every 5 seconds.");
}

void loop() {
  // Read data from the sensors
  int sensorValues1[NUM_SENSORS];
  int sensorValues2[NUM_SENSORS];
  
  for (int i = 0; i < NUM_SENSORS; i++) {
    sensorValues1[i] = analogRead(SENSOR_PIN1);
    sensorValues2[i] = analogRead(SENSOR_PIN2);
  }

  // Prepare data to broadcast
  char data[512]; // Increased size for combined data
  snprintf(data, sizeof(data), 
           "A1:%d, A2:%d, A3:%d, A4:%d, A5:%d, A6:%d, A7:%d, A8:%d, A9:%d, A10:%d, A11:%d, A12:%d, A13:%d, A14:%d, A15:%d, B1:%d, B2:%d, B3:%d, B4:%d, B5:%d, B6:%d, B7:%d, B8:%d, B9:%d, B10:%d, B11:%d, B12:%d, B13:%d, B14:%d, B15:%d",
           sensorValues1[0], sensorValues1[1], sensorValues1[2], sensorValues1[3], sensorValues1[4], sensorValues1[5], sensorValues1[6], sensorValues1[7], sensorValues1[8], sensorValues1[9], sensorValues1[10], sensorValues1[11], sensorValues1[12], sensorValues1[13], sensorValues1[14],
           sensorValues2[0], sensorValues2[1], sensorValues2[2], sensorValues2[3], sensorValues2[4], sensorValues2[5], sensorValues2[6], sensorValues2[7], sensorValues2[8], sensorValues2[9], sensorValues2[10], sensorValues2[11], sensorValues2[12], sensorValues2[13], sensorValues2[14]);

  Serial.printf("Broadcasting message: %s\n", data);

  // Broadcast the sensor data to all devices within the network
  if (!broadcast_peer.send_message((uint8_t *)data, sizeof(data))) {
    Serial.println("Failed to broadcast message");
  }

  delay(500);  // Wait before the next broadcast
}
