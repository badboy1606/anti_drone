#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "driver/ledc.h"
#include "driver/uart.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define SERVO_X_PIN 18
#define SERVO_Y_PIN 19

#define SERVO_MIN_US 500
#define SERVO_MAX_US 2500
#define SERVO_FREQ   50

char line[32];
int idx = 0;

// Convert angle → PWM duty
void set_servo_angle(int gpio, int angle)
{
    if (angle < 0) angle = 0;
    if (angle > 180) angle = 180;

    int pulse = SERVO_MIN_US +
                (SERVO_MAX_US - SERVO_MIN_US) * angle / 180;

    uint32_t duty = (pulse * 8191) / (1000000 / SERVO_FREQ);

    ledc_set_duty(LEDC_LOW_SPEED_MODE,
                  gpio == SERVO_X_PIN ? LEDC_CHANNEL_0 : LEDC_CHANNEL_1,
                  duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE,
                     gpio == SERVO_X_PIN ? LEDC_CHANNEL_0 : LEDC_CHANNEL_1);
}

void app_main(void)
{
    // --- LEDC Servo Setup ---
    ledc_timer_config_t timer = {
        .speed_mode      = LEDC_LOW_SPEED_MODE,
        .duty_resolution = LEDC_TIMER_13_BIT,
        .timer_num       = LEDC_TIMER_0,
        .freq_hz         = SERVO_FREQ,
        .clk_cfg         = LEDC_AUTO_CLK
    };
    ledc_timer_config(&timer);

    ledc_channel_config_t ch0 = {
        .gpio_num   = SERVO_X_PIN,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel    = LEDC_CHANNEL_0,
        .timer_sel  = LEDC_TIMER_0
    };
    ledc_channel_config(&ch0);

    ledc_channel_config_t ch1 = {
        .gpio_num   = SERVO_Y_PIN,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel    = LEDC_CHANNEL_1,
        .timer_sel  = LEDC_TIMER_0
    };
    ledc_channel_config(&ch1);

    // --- UART0 setup (USB cable) ---
    uart_config_t cfg = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .stop_bits = UART_STOP_BITS_1,
        .parity    = UART_PARITY_DISABLE,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE
    };

    uart_param_config(UART_NUM_0, &cfg);

    // IMPORTANT: ESP-IDF 5.2 needs 6 args
    uart_driver_install(UART_NUM_0, 1024, 0, 0, NULL, 0);

    uint8_t ch;

    while (1)
    {
        if (uart_read_bytes(UART_NUM_0, &ch, 1, 10 / portTICK_PERIOD_MS) > 0)
        {
            if (ch == '\n')
            {
                line[idx] = '\0';

                char axis = line[0];
                int angle = atoi(&line[1]);

                if (axis == 'X')
                    set_servo_angle(SERVO_X_PIN, angle);

                else if (axis == 'Y')
                    set_servo_angle(SERVO_Y_PIN, angle);

                idx = 0;
            }
            else
            {
                line[idx++] = ch;
                if (idx >= sizeof(line)) idx = 0;
            }
        }
    }
}
