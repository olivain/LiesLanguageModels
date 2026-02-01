#include <Arduino.h>
#include <GxEPD2_BW.h>
#include <gdey/GxEPD2_370_GDEY037T03.h>
#define EPD_CS 15
#define EPD_DC 4
#define EPD_RST 2
#define EPD_BUSY 5

#define GxEPD2_DISPLAY_CLASS GxEPD2_BW
#define GxEPD2_DRIVER_CLASS GxEPD2_370_GDEY037T03

#define MAX_DISPLAY_BUFFER_SIZE 12480ul
#define MAX_HEIGHT(EPD) (EPD::HEIGHT <= MAX_DISPLAY_BUFFER_SIZE / (EPD::WIDTH / 8) ? EPD::HEIGHT : MAX_DISPLAY_BUFFER_SIZE / (EPD::WIDTH / 8))

GxEPD2_DISPLAY_CLASS<GxEPD2_DRIVER_CLASS, MAX_HEIGHT(GxEPD2_DRIVER_CLASS)> display(GxEPD2_DRIVER_CLASS(EPD_CS, EPD_DC, EPD_RST, EPD_BUSY));

unsigned char imageBuffer[MAX_DISPLAY_BUFFER_SIZE];

enum State
{
    WAIT_CMD_OR_LEN,
    PULSING,
    RECV_IMAGE
};
State state = WAIT_CMD_OR_LEN;

bool pulseColor = false;
uint32_t last_pulse = millis();

static void pulseOnce()
{
    display.setPartialWindow(0, 0, display.width(), display.height());
    display.firstPage();
    do
    {
        yield();
        display.fillScreen(pulseColor ? GxEPD_BLACK : GxEPD_WHITE);
    } while (display.nextPage());
    pulseColor = !pulseColor;
}

void displayString(String ipText)
{
    display.setFullWindow();
    display.firstPage();
    do
    {
        yield();
        display.fillScreen(GxEPD_WHITE);
        display.setCursor(0, 20);
        display.setTextColor(GxEPD_BLACK);
        display.setFont(NULL);
        display.setTextSize(5);
        display.print(ipText);
    } while (display.nextPage());

    display.powerOff();
}

static bool readExact(uint8_t *dst, uint32_t n, uint32_t timeoutMs)
{
    uint32_t got = 0;
    uint32_t last = millis();
    while (got < n)
    {
        while (Serial.available() && got < n)
        {
            dst[got++] = (uint8_t)Serial.read();
            last = millis();
            yield();
        }
        if (millis() - last > timeoutMs)
            return false;
        yield();
    }
    return true;
}

void setup()
{
    Serial.begin(115200); // match Python
    Serial.setTimeout(50);

    display.init();
    display.setRotation(2);

    displayString("LIES\nLANGUAGE\nMODELS\n\nOLIVAIN\nPORRY\n2026");
}

void loop()
{
    static uint32_t expectedLen = 0;
    static uint32_t received = 0;

    // Command: PULSE
    if (state == WAIT_CMD_OR_LEN && Serial.available() >= 5 && Serial.peek() == 'P')
    {
        char b[6] = {0};
        Serial.readBytes(b, 5);
        if (strcmp(b, "PULSE") == 0)
        {
            Serial.println("OK\n");
            Serial.flush(); // waits until TX buffer is actually sent
            
            state = PULSING;

            return;
        }
    }

    if (state == PULSING)
    {
        // If host started sending a length header, stop pulsing immediately
        if (Serial.available() >= 4 && Serial.peek() != 'P')
        {
            // Read the 4-byte big-endian length right now
            uint8_t lenBytes[4];
            for (int i = 0; i < 4; i++)
            {
                lenBytes[i] = (uint8_t)Serial.read();
                yield();
            }

            uint32_t expectedLen =
                (uint32_t(lenBytes[0]) << 24) |
                (uint32_t(lenBytes[1]) << 16) |
                (uint32_t(lenBytes[2]) << 8) |
                (uint32_t(lenBytes[3]) << 0);

            if (expectedLen == 0 || expectedLen > MAX_DISPLAY_BUFFER_SIZE)
            {
                Serial.print("ERR_LEN\n");
                // Resync back to command mode
                while (Serial.available())
                    Serial.read();
                state = WAIT_CMD_OR_LEN;
                return;
            }

            // Tell Python we’re ready to receive payload bytes
            Serial.print("READY\n");

            // Receive payload bytes (blocking with timeout)
            uint32_t received = 0;
            uint32_t last = millis();
            while (received < expectedLen)
            {
                while (Serial.available() && received < expectedLen)
                {
                    imageBuffer[received++] = (uint8_t)Serial.read();
                    last = millis();
                    yield();
                }
                if (millis() - last > 2000)
                {
                    Serial.print("ERR_TIMEOUT\n");
                    while (Serial.available())
                        Serial.read();
                    state = WAIT_CMD_OR_LEN;
                    return;
                }
                yield();
            }

            // Draw
            display.setFullWindow();
            display.firstPage();
            do
            {
                yield();
                display.fillScreen(GxEPD_WHITE);
                display.drawInvertedBitmap(0, 0, imageBuffer, display.width(), display.height(), GxEPD_BLACK);
            } while (display.nextPage());
            display.powerOff();

            Serial.print("DONE\n");
            state = WAIT_CMD_OR_LEN;
            return;
        }

        // Otherwise keep pulsing
        display.setPartialWindow(0, 0, display.width(), display.height());
        display.firstPage();
        do
        {
            yield();
            display.fillScreen(pulseColor ? GxEPD_BLACK : GxEPD_WHITE);
        } while (display.nextPage());
        pulseColor = !pulseColor;

        delay(50);
        return;
    }

    // If we weren't pulsing, we can also accept length header directly
    if (state == WAIT_CMD_OR_LEN)
    {
        if (Serial.available() >= 4 && Serial.peek() != 'P')
        {
            uint8_t lenBytes[4];
            if (!readExact(lenBytes, 4, 500))
                return;

            expectedLen = (uint32_t(lenBytes[0]) << 24) |
                          (uint32_t(lenBytes[1]) << 16) |
                          (uint32_t(lenBytes[2]) << 8) |
                          (uint32_t(lenBytes[3]) << 0);

            if (expectedLen == 0 || expectedLen > MAX_DISPLAY_BUFFER_SIZE)
            {
                Serial.print("ERR_LEN\n");
                expectedLen = 0;
                return;
            }

            received = 0;
            state = RECV_IMAGE;
            return;
        }
        return;
    }

    if (state == RECV_IMAGE)
    {
        // Read remaining bytes with timeout
        while (Serial.available() && received < expectedLen)
        {
            imageBuffer[received++] = (uint8_t)Serial.read();
            yield();
        }

        static uint32_t lastProgress = 0;
        if (received == 0)
            lastProgress = millis();
        if (Serial.available())
            lastProgress = millis();

        if (received < expectedLen && (millis() - lastProgress) > 2000)
        {
            Serial.print("ERR_TIMEOUT\n");
            // reset
            state = WAIT_CMD_OR_LEN;
            expectedLen = 0;
            received = 0;
            while (Serial.available())
                Serial.read();
            return;
        }

        if (received == expectedLen)
        {
            display.setFullWindow();
            display.firstPage();
            do
            {
                yield();
                display.fillScreen(GxEPD_WHITE);
                display.drawInvertedBitmap(0, 0, imageBuffer, display.width(), display.height(), GxEPD_BLACK);
            } while (display.nextPage());
            display.powerOff();

            Serial.print("DONE\n");

            state = WAIT_CMD_OR_LEN;
            expectedLen = 0;
            received = 0;
            return;
        }
    }
}
