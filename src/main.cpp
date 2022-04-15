/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <FlexCAN_T4.h>
#include <TensorFlowLite.h>

#include "main_functions.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <Arduino.h>

#define encoder A0
#define emg1 A1
#define emg2 A2

FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;
CAN_message_t msgsend;
CAN_message_t msg;

// EMG stuff
const float EMG_GAIN = pow(10, -7);

// Inference
float infer;
float infer_prev;
int output_final;
int flag = 0;

// Raw data array
float emgdata[1][9][3];


// Input array
void input_array_update(){
	
  for (int i =  0; i < 9; i++) {
    //emgdata[0][i][0] = analogRead(emg1)*EMG_GAIN;
	//emgdata[0][i][1] = analogRead(emg2)*EMG_GAIN;
	emgdata[0][i][0] = analogRead(emg1);
	emgdata[0][i][1] = analogRead(emg2);
	emgdata[0][i][2] = analogRead(encoder);
	
    delay(10);
  }
	
}

/*
void copy_buffer_to_tensor(float *input_tensor) {
	
	for (int i=0; i<9; i++){
		input_tensor[0][i][0] = emgdata[0][i][0];
		input_tensor[0][i][1] = emgdata[0][i][1];
		input_tensor[0][i][2] = emgdata[0][i][2];
	}
}
*/

void copy_buffer_to_tensor_row_major(float *input_tensor) {
	
	for (int i=0; i<9; i++){
		input_tensor[0 + (3*i)] = emgdata[0][i][0];
		input_tensor[1 + (3*i)] = emgdata[0][i][1];
		input_tensor[2 + (3*i)] = emgdata[0][i][2];
	}
}


void copy_buffer_to_tensor_column_major(float *input_tensor) {

    for (byte i=0; i<9; i++) {
		input_tensor[i] = emgdata[0][i][0];
		input_tensor[i + 9] = emgdata[0][i][1];
		input_tensor[i + (9*2)] = emgdata[0][i][1];
    }
}

void canSniff20(const CAN_message_t &msg) { // global callback  // Whenever receive msg, will run this //like "can_interrupt_handler" in mbed code

  if (msg.len ==0 ){ //(msg.id == 0x5A){
  
  //***** Encoder *****//
  int sensorValue0 = analogRead(encoder);
  Serial.println(sensorValue0);
  
  int n = sensorValue0;
  byte upper =  n >> 8;

  if (flag == 1){

	  if (output_final == 1){
		  upper = upper || 0x04;
	  }
	  else if (output_final == -1){
		  upper = upper || 0x08;
	  }
	  else{
		  upper = upper || 0x00;
	  }
	  flag = 0;
  }
  byte lower =  n & 0xFF;
  byte analogdata[2] = {upper,lower};

  // teensy id 0 01110 11000 = 1D8 means (14 then 24) TEENSYID = 14; TEENSY_REPLY_1  = 24; (CMD ID) all self-define
  msgsend.id = 0x1D8;
  msgsend.len = 2; 
  msgsend.buf[0] = analogdata[0];
  msgsend.buf[1] = analogdata[1];

  //Serial.println(analogdata[0]);
  //Serial.println(analogdata[1]);

  //***** CAN *****//
  can1.write(msgsend);
  
  }
}


// Globals, used for compatibility with Arduino-style sketches.
// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 1*1024*3;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


void setup(void) {
	
	
	Serial.begin(115200);
	delay(1000);
	Serial.println("Teensy 4.1 CAN test");

	can1.begin();
	can1.setBaudRate(1000000); // MUST BE SAME AS SENSOR

	can1.enableFIFO();
	can1.enableFIFOInterrupt();
	can1.onReceive(FIFO, canSniff20); // allows FIFO messages to be received in the supplied callback. (function as argument)
  
	tflite::InitializeTarget();
	// Set up logging. Google style is to avoid globals or statics because of
	// lifetime uncertainty, but since this has a trivial destructor it's okay.
	// NOLINTNEXTLINE(runtime-global-variables)
	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;
  
	// Map the model into a usable data structure. This doesn't involve any
	// copying or parsing, it's a very lightweight operation.
	model = tflite::GetModel(g_model);
	
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
		return;
	}
  
	// This pulls in all the operation implementations we need.
	// NOLINTNEXTLINE(runtime-global-variables)
	static tflite::AllOpsResolver resolver;  
  
	// Build an interpreter to run the model with.
	static tflite::MicroInterpreter static_interpreter(
		model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
	interpreter = &static_interpreter;
	
	// Allocate memory from the tensor_arena for the model's tensors.
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
		return;
	}
	
	// Obtain pointers to the model's input and output tensors.
	input = interpreter->input(0);
	output = interpreter->output(0);
	
	// Keep track of how many inferences we have performed.
	inference_count = 0;

	delay(1000);
	
	// OUTPUT INITIALIZATION
	infer_prev = analogRead(encoder);

}

void loop() {

	can1.events(); // ensure program is running; actually can leave the loop empty. 
	// Update the EMG input Buffer
	input_array_update();
	// Place the input in the model's input tensor
	copy_buffer_to_tensor_row_major(input->data.f);
	
	// Run inference, and report any error
	TfLiteStatus invoke_status = interpreter->Invoke();
	if (invoke_status != kTfLiteOk) {
		TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n");
		//return;
	}
	
	/********************** INEFERENCE ENDS HERE ************************/
	
	//infer = output->data.f[0][0][0];
	infer = output->data.f[0];

	if (infer - infer_prev > 1){
		output_final = -1;		// extension
	}
	else if (infer - infer_prev < -1){
		output_final = 1;		// flexion
	}
	else{
		output_final = 0;		// no motion
	}
	
	Serial.print(output_final);
	flag = 1;


}