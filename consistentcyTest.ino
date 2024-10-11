// Define ADC pins
#define ADC_PIN1 12
#define ADC_PIN2 13
#define ADC_PIN3 25                                                      
#define ADC_PIN4 33
#define ADC_PIN5 21
#define ADC_PIN6 19

// Define the output pin


void setup() {
  // Initialize serial communication for debugging
  Serial.begin(921600);
  
  // Initialize ADC pins
  pinMode(ADC_PIN1, INPUT);
  pinMode(ADC_PIN2, INPUT);
  pinMode(ADC_PIN3, INPUT);
  pinMode(ADC_PIN4, INPUT);
  pinMode(ADC_PIN5, INPUT);
  pinMode(ADC_PIN6, INPUT);

  // Initialize the output pin

}

void loop() {
  // Read analog values from ADC pins
  int sensorValue1 = analogRead(ADC_PIN1);
  int sensorValue2 = analogRead(ADC_PIN2);
  int sensorValue3 = analogRead(ADC_PIN3);
  int sensorValue4 = analogRead(ADC_PIN4);
  int sensorValue5 = analogRead(ADC_PIN5);
  int sensorValue6 = analogRead(ADC_PIN6);

  // Prepare the output string
  String outputData = "A1:" + String(sensorValue1) + ", " +
                      "A2:" + String(sensorValue2) + ", " +
                      "A3:" + String(sensorValue3)  + ", " +
                      "A4:" + String(sensorValue4) + ", " +
                      "A5:" + String(sensorValue5) + ", " +
                      "A6:" + String(sensorValue6);
 
  // Print the output data to serial (for debugging)
  Serial.println(outputData);

  // Send the output data to GPIO 23 (for another ESP32 to read)
  // sendToGPIO23(outputData);

  // Wait before next reading
  delay(100);
}

// void sendToGPIO23(String data) {
//   for (int i = 0; i < data.length(); i++) {
//     // Write each character of the string to the output pin
//     digitalWrite(OUTPUT_PIN, data[i]);
//     delay(10);  // Small delay for reliable transmission
//   }}
