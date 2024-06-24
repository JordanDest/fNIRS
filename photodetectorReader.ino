// Define the analog pins for the photodiode anode and the resistor junction
const int photodiodeAnodePin = A0;
const int resistorEndPin = A1;

// Variables to store the analog values
int photodiodeAnodeValue = 0;
int resistorEndValue = 0;

void setup() {
  // Start the serial communication
  Serial.begin(9600);

  // Set the analog pins as input
  pinMode(photodiodeAnodePin, INPUT);
  pinMode(resistorEndPin, INPUT);

  // Print test messages to check if A0 and A1 are recognized
  Serial.print("A0: ");
  Serial.println(A0);
  Serial.print("A1: ");
  Serial.println(A1);
}

void loop() {
  // Read the analog values from the photodiode anode and resistor end
  photodiodeAnodeValue = analogRead(photodiodeAnodePin);
  resistorEndValue = analogRead(resistorEndPin);

  // Print the values to the Serial Monitor
  Serial.print("Photodiode Anode Value: ");
  Serial.print(photodiodeAnodeValue);
  Serial.print("\tResistor End Value: ");
  Serial.println(resistorEndValue);

  // Add a small delay to avoid overwhelming the Serial Monitor
  delay(500);
}
