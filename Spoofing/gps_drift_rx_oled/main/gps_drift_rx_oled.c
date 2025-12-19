#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "GPS_RX";

typedef struct {
    float lat;
    float lon;
    float alt;
} gps_packet_t;

gps_packet_t incoming = {0};
bool first_signal_received = false;
bool started = false;

#define START_LAT 19.0
#define START_LON 72.0
#define START_ALT 0.0

// ---------------------- RSSI TAKEOVER ADDITIONS ----------------------------
typedef struct {
    uint8_t mac[6];
    float ema;
    bool valid;
} mac_rssi_t;

mac_rssi_t sat = { .valid = false };
mac_rssi_t spoof = { .valid = false };

#define ALPHA 0.25f
#define SWITCH_MARGIN 6.0f
#define MIN_PACKETS_BEFORE_SWITCH 3

int spoof_count = 0;

// promiscuous callback ONLY updates RSSI, no MAC assignment here anymore
static void promisc_cb(void *buf, wifi_promiscuous_pkt_type_t type)
{
    const wifi_promiscuous_pkt_t *p = (wifi_promiscuous_pkt_t *)buf;
    const uint8_t *raw = p->payload;
    if (!raw) return;
    if (p->rx_ctrl.sig_len < 24) return;

    uint8_t src[6];
    memcpy(src, raw + 10, 6);
    int rssi = p->rx_ctrl.rssi;

    // update SAT RSSI
    if (sat.valid && memcmp(src, sat.mac, 6) == 0) {
        sat.ema = ALPHA * rssi + (1 - ALPHA) * sat.ema;
        return;
    }

    // update SPOOF RSSI
    if (spoof.valid && memcmp(src, spoof.mac, 6) == 0) {
        spoof.ema = ALPHA * rssi + (1 - ALPHA) * spoof.ema;
        return;
    }
}
// ---------------------------------------------------------------------------

// ---------------------- CHANNEL FIX TO 1 ----------------------------
int current_channel = 1;
bool channel_locked = false;
// -------------------------------------------------------------------

void display_coordinates(float lat, float lon, float alt)
{
    ESP_LOGI(TAG, "Lat: %.6f | Lon: %.6f | Alt: %.1f", lat, lon, alt);
}

static void recv_cb(const esp_now_recv_info_t *info, const uint8_t *data, int len)
{
    if (!channel_locked) {
        channel_locked = true;
        ESP_LOGI(TAG, "ESP-NOW signal detected! LOCKED to channel 1");
    }

    // ----------- REGISTER SATELLITE MAC ONLY ON FIRST ESP-NOW PACKET -----------
    if (!sat.valid) {
        memcpy(sat.mac, info->src_addr, 6);
        sat.ema = -50;
        sat.valid = true;

        ESP_LOGI(TAG, "REGISTERED SATELLITE MAC: %02X:%02X:%02X:%02X:%02X:%02X",
                 sat.mac[0], sat.mac[1], sat.mac[2],
                 sat.mac[3], sat.mac[4], sat.mac[5]);
    }
    // ----------------------------------------------------------------------------

    // Assign spoofer MAC when a NEW ESP-NOW sender appears
    if (sat.valid && !spoof.valid &&
        memcmp(info->src_addr, sat.mac, 6) != 0)
    {
        memcpy(spoof.mac, info->src_addr, 6);
        spoof.ema = -50;
        spoof.valid = true;

        ESP_LOGI(TAG, "REGISTERED SPOOFER MAC: %02X:%02X:%02X:%02X:%02X:%02X",
                 spoof.mac[0], spoof.mac[1], spoof.mac[2],
                 spoof.mac[3], spoof.mac[4], spoof.mac[5]);
    }

    // ---------------------- TAKEOVER LOGIC ----------------------------
    if (spoof.valid && sat.valid) {
        if (spoof.ema > sat.ema + SWITCH_MARGIN) {
            spoof_count++;
            if (spoof_count >= MIN_PACKETS_BEFORE_SWITCH) {
                ESP_LOGW(TAG, "TAKEOVER: Spoofer stronger (%.1f dB > %.1f dB)",
                         spoof.ema, sat.ema);
            }
        } else {
            spoof_count = 0;
        }
    }
    // -----------------------------------------------------------------

    if (!first_signal_received) {
        first_signal_received = true;
        ESP_LOGI(TAG, "First ESP-NOW packet source registered");
        vTaskDelay(7000 / portTICK_PERIOD_MS);
        started = true;
        return;
    }

    if (!started) return;
    if (len != sizeof(gps_packet_t)) return;

    memcpy(&incoming, data, sizeof(incoming));
    display_coordinates(incoming.lat, incoming.lon, incoming.alt);
}

void app_main()
{
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    // ---------------------- FORCE RECEIVER TO CHANNEL 1 ----------------------
    esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
    ESP_LOGI(TAG, "Receiver forced to channel 1");
    // -------------------------------------------------------------------------

    esp_wifi_set_promiscuous(true);
    esp_wifi_set_promiscuous_rx_cb(promisc_cb);

    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
    ESP_LOGI(TAG, "Receiver MAC: %02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    // Display starting coordinates
    display_coordinates(START_LAT, START_LON, START_ALT);

    ESP_ERROR_CHECK(esp_now_init());
    esp_now_register_recv_cb(recv_cb);

    ESP_LOGI(TAG, "Ready...");
}