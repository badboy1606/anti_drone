#include <stdio.h>
#include <string.h>
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "SAT_TX";

typedef struct {
    float lat;
    float lon;
    float alt;
} gps_packet_t;

// Starting coordinates for REAL satellite
gps_packet_t gps_data = {
    .lat = 19.000000,
    .lon = 72.000000,
    .alt = 10.0,
};

void send_cb(const uint8_t *mac, esp_now_send_status_t status)
{
    ESP_LOGI(TAG, "Send status: %s", status == ESP_NOW_SEND_SUCCESS ? "SUCCESS" : "FAIL");
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

    // *********** FORCE WIFI CHANNEL 1 ***********
    ESP_ERROR_CHECK(esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE));
    ESP_LOGI(TAG, "Satellite forced to Wi-Fi Channel 1");
    // *********************************************

    // Print MAC (optional, but left unchanged)
    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
    ESP_LOGI(TAG, "Satellite ESP MAC: %02X:%02X:%02X:%02X:%02X:%02X",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);

    // Init ESP-NOW
    ESP_ERROR_CHECK(esp_now_init());
    esp_now_register_send_cb(send_cb);

    // *********** BROADCAST PEER, NO CHANNEL SET ***********
    esp_now_peer_info_t peer = {0};
    memset(&peer, 0, sizeof(peer));

    // broadcast MAC
    memcpy(peer.peer_addr, "\xFF\xFF\xFF\xFF\xFF\xFF", 6);

    peer.ifidx = ESP_IF_WIFI_STA;
    peer.encrypt = false;
    // peer.channel = NOT SET (default behavior)
    // *******************************************************

    if (esp_now_add_peer(&peer) != ESP_OK) {
        ESP_LOGW(TAG, "Peer already exists or error occurred");
    }

    ESP_LOGI(TAG, "=== REAL SATELLITE TRAJECTORY ===");
    ESP_LOGI(TAG, "Starting from: Lat %.6f, Lon %.6f", gps_data.lat, gps_data.lon);
    ESP_LOGI(TAG, "Increment: +0.05 per transmission");
    ESP_LOGI(TAG, "=================================");

    // Transmit loop with incrementing coordinates
    while (1) {
        // Send current GPS data
        esp_now_send(peer.peer_addr, (uint8_t *)&gps_data, sizeof(gps_data));

        ESP_LOGI(TAG, "Sent REAL GPS: Lat %.6f | Lon %.6f | Alt %.1f",
                 gps_data.lat, gps_data.lon, gps_data.alt);

        // Increment coordinates by 0.4 for next transmission
        gps_data.lat += 0.05;
        gps_data.lon += 0.05;

        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
}