// Define ADC pins for B1-B15
//Left Sideburn
#define B1 2
#define B2 4
#define B3 15
#define B4 36
//Left Side of 8
#define B5 27
#define B6 33
#define B7 32
#define B8 39
#define B9 26
#define B10 25
#define B11 35
#define B12 34
//Center
#define B13 14
#define B14 12
#define B15 13

// Define the output pin
//#define OUTPUT_PIN 23

void setup() {
  // Initialize serial communication for debugging
  Serial.begin(115200);
  
  // Initialize ADC pins
  pinMode(B1, INPUT);
  pinMode(B2, INPUT);
  pinMode(B3, INPUT);
  pinMode(B4, INPUT);
  pinMode(B5, INPUT);
  pinMode(B6, INPUT);
  pinMode(B7, INPUT);
  pinMode(B8, INPUT);
  pinMode(B9, INPUT);
  pinMode(B10, INPUT);
  pinMode(B11, INPUT);
  pinMode(B12, INPUT);
  pinMode(B13, INPUT);
  pinMode(B14, INPUT);
  pinMode(B15, INPUT);

  // Initialize the output pin
  //pinMode(OUTPUT_PIN, OUTPUT);
}

void loop() {
  // Read analog values from ADC pins
  // int sensorValues[15];
  int sensorValues[8];
  sensorValues[0] = analogRead(B1);
  sensorValues[1] = analogRead(B2);
  sensorValues[2] = analogRead(B3);
  sensorValues[3] = analogRead(B4);
  sensorValues[4] = analogRead(B5);
  sensorValues[5] = analogRead(B6);
  sensorValues[6] = analogRead(B7);
  sensorValues[7] = analogRead(B8);
  sensorValues[8] = analogRead(B9);
  sensorValues[9] = analogRead(B10);
  sensorValues[10] = analogRead(B11);
  sensorValues[11] = analogRead(B12);
  sensorValues[12] = analogRead(B13);
  sensorValues[13] = analogRead(B14);
  sensorValues[14] = analogRead(B15);

  // Prepare the output string
   String outputData = "B1:" + String(sensorValues[0]) + ", " +
                      "B2:" + String(sensorValues[1]) + ", " +
                      "B3:" + String(sensorValues[2]) + ", " +
                      "B4:" + String(sensorValues[3]) + ", " +
                      "B5:" + String(sensorValues[4]) + ", " +
                      "B6:" + String(sensorValues[5]) + ", " +
                      "B7:" + String(sensorValues[6]) + ", " +
                      "B8:" + String(sensorValues[7]) + ", " +
                      "B9:" + String(sensorValues[8]) + ", " +
                      "B10:" + String(sensorValues[9]) + ", " +
                      "B11:" + String(sensorValues[10]) + ", " +
                      "B12:" + String(sensorValues[11]) + ", " +
                      "B13:" + String(sensorValues[12]) + ", " +
                       "B14:" + String(sensorValues[13]) + ", " +
                       "B15:" + String(sensorValues[14]);

  // Print the output data to serial (for debugging)
  Serial.println(outputData);

  // Send the output data to GPIO 23 (for another ESP32 to read)
  //sendToGPIO23(outputData);

  // Wait before next reading
  delay(100);
}

// void sendToGPIO23(String data) {
//   for (int i = 0; i < data.length(); i++) {
//     // Write each character of the string to the output pin
//     digitalWrite(OUTPUT_PIN, data[i]);
//     delay(10);  // Small delay for reliable transmission
//   }
// }
