// Define ADC pins

// Right Sideburn
#define A1 2
#define A2 4
#define A3 15
#define A4 39
//Right Plate of 8
#define A5 32
#define A6 35
#define A7 34
#define A8 36
#define A9 33
#define A10 27
#define A11 26
#define A12 25
// Center
#define A13 14
#define A14 13
#define A15 12

// Define the output pin
#define OUTPUT_PIN 23

void setup() {
  // Initialize serial communication for debugging
  Serial.begin(115200);
  
  // Initialize ADC pins
  pinMode(A1, INPUT);
  pinMode(A2, INPUT);
  pinMode(A3, INPUT);
  pinMode(A4, INPUT);
  pinMode(A5, INPUT);
  pinMode(A6, INPUT);
  pinMode(A7, INPUT);
  pinMode(A8, INPUT);
  pinMode(A9, INPUT);
  pinMode(A10, INPUT);
  pinMode(A11, INPUT);
  pinMode(A12, INPUT);
  pinMode(A13, INPUT);
  pinMode(A14, INPUT);
  pinMode(A15, INPUT);

  // Initialize the output pin
  pinMode(OUTPUT_PIN, OUTPUT);
}

void loop() {
  // Read analog values from ADC pins
  //int sensorValues[15];
  int sensorValues[8];
  sensorValues[0] = analogRead(A1);
  sensorValues[1] = analogRead(A2);
  sensorValues[2] = analogRead(A3);
  sensorValues[3] = analogRead(A4);
  sensorValues[4] = analogRead(A5);
  sensorValues[5] = analogRead(A6);
  sensorValues[6] = analogRead(A7);
  sensorValues[7] = analogRead(A8);
  sensorValues[8] = analogRead(A9);
  sensorValues[9] = analogRead(A10);
  sensorValues[10] = analogRead(A11);
  sensorValues[11] = analogRead(A12);
  sensorValues[12] = analogRead(A13);
  sensorValues[13] = analogRead(A14);
  sensorValues[14] = analogRead(A15);

  //Prepare the output string
  String outputData = 
                     "A1:" + String(sensorValues[0]) + ", " +
                      "A2:" + String(sensorValues[1]) + ", " +
                      "A3:" + String(sensorValues[2]) + ", " +
                      "A4:" + String(sensorValues[3]) + ", " +
                      "A5:" + String(sensorValues[4]) + ", " +
                      "A6:" + String(sensorValues[5]) + ", " +
                      "A7:" + String(sensorValues[6]) + ", " +
                      "A8:" + String(sensorValues[7])  + ", " +
                      "A9:" + String(sensorValues[8]) + ", " +
                      "A10:" + String(sensorValues[9]) + ", " +
                      "A11:" + String(sensorValues[10]) + ", " +
                      "A12:" + String(sensorValues[11]) + ", " +
                      "A13:" + String(sensorValues[12]) + ", " +
                      "A14:" + String(sensorValues[13]) + ", " +
                      "A15:" + String(sensorValues[14]);

  // Print the output data to serial (for debugging)
  Serial.println(outputData);

  // Send the output data to GPIO 23 (for another ESP32 to read)
  sendToGPIO23(outputData);

  // Wait before next reading
  delay(1);
}

void sendToGPIO23(String data) {
  for (int i = 0; i < data.length(); i++) {
    // Write each character of the string to the output pin
    digitalWrite(OUTPUT_PIN, data[i]);
    delay(10);  // Small delay for reliable transmission
  }
}
