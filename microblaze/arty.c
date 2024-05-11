/******************************************************************************
*
* Copyright (C) 2009 - 2014 Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* Use of the Software is limited solely to applications:
* (a) running on a Xilinx device, or
* (b) that interact with a Xilinx device through a bus or interconnect.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in
* this Software without prior written authorization from Xilinx.
*
******************************************************************************/

/*
 * helloworld.c: simple test application
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */

#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include "xtracking.h"

XTracking do_xtracking;
XTracking_Config *xtracking_cfg;

void init_HLS_tracking();
void tracking(float observations[12], int timestep);

int main()
{
    init_platform();
    init_HLS_tracking();
    float observations[12] = {0};
    int timestep;
    while (1) {
		int num = scanf("%f %f %f %f %f %f %f %f %f %f %f %f %d", &observations[0],  &observations[1],
				 &observations[2],  &observations[3],  &observations[4],  &observations[5],
				 &observations[6],  &observations[7],  &observations[8],  &observations[9],
				 &observations[10],  &observations[11], &timestep);
		if (num != 13) {
			while (getchar() != '\n') continue;
			continue;
		}
		tracking(observations, timestep);
    }

    cleanup_platform();
    return 0;
}

void init_HLS_tracking(){

		int status;
		// Create HLS example IP pointer
		xtracking_cfg = XTracking_LookupConfig(
		XPAR_XTRACKING_0_DEVICE_ID);

		if (!xtracking_cfg) {
//			xil_printf(
//					"Error loading configuration for do_hls_example_cfg \n\r");
			print("config error \n\r");
		}


		status = XTracking_CfgInitialize(&do_xtracking,
				xtracking_cfg);
		if (status != XST_SUCCESS) {
//			xil_printf("Error initializing for do_hls_example \n\r");
			print("config error 2 \n\r");
		}

		XTracking_Initialize(&do_xtracking,
		XPAR_XTRACKING_0_DEVICE_ID);
}

void tracking(float observations[12], int timestep) {
	uint32_t inputs[4] = {0};

	// Write inputs
	XTracking_Write_observations_Bytes(&do_xtracking, 0, (char*)observations, 12*4);
	XTracking_Set_timestep(&do_xtracking, timestep);
//	printf("Write observations: %f %f %f ... \n\r", observations[0], observations[1], observations[2]);

	// Start HLS IP
	while (!XTracking_IsReady(&do_xtracking)) {
//		print("waiting for Xtracking to be ready\n\r");
	}
	XTracking_Start(&do_xtracking);
//	xil_printf("Started HLS Tracking IP \n\r");

	// Wait until it is finished
	while (!XTracking_IsDone(&do_xtracking));

	// Get hls_multiplier returned value
	XTracking_Read_inputs_Bytes(&do_xtracking, 0, (char*)inputs, 4*4);
	xil_printf("%u %u %u %u\n", inputs[0], inputs[1], inputs[2], inputs[3]);
//	xil_printf("End of function \n\n\r");

//	print("end of tracking\n\r");
}
