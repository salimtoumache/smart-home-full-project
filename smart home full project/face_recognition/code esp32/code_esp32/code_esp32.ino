
#include <ESP32Servo.h>
Servo MAIN_DOOR;
String myCMD;

void setup() {
  Serial.begin(9600);
  MAIN_DOOR.attach(15);
}

void loop() {
  SERIAL_PYTHON();
}
// SERIAL PYTHON
void SERIAL_PYTHON() {
  while (Serial.available() == 0) {
  }
  myCMD = Serial.readStringUntil('\r');
  if (myCMD == "ON") {
    MAIN_DOOR.write(180);
  }
  else if (myCMD == "OFF") {
    MAIN_DOOR.write(0);
  }
}
