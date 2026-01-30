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
    WAIT_HEADER,
    PULSING,
    WAIT_IMAGE
};

State state = WAIT_HEADER;

bool pulseColor = true; // true = white, false = black
// Add these to your global variables at the top
int scanY = 0;
const int scanHeight = 50; // Height of the "beam"

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

void setup()
{
    delay(100);
    Serial.begin(230400);
    display.init();
    display.setRotation(2);

    delay(100);

    displayString("LIES\nLANGUAGE\nMODELS\n\nOLIVAIN\nPORRY\n2026");
}

void loop()
{

    static uint32_t expectedLength = 0;
    static uint32_t receivedBytes = 0;

    // 1. Check for the "PULSE" command string
    if (state == WAIT_HEADER && Serial.available() >= 5)
    {
        if (Serial.peek() == 'P')
        {
            char buffer[6] = {0};
            Serial.readBytes(buffer, 5);
            if (strcmp(buffer, "PULSE") == 0)
            {
                state = PULSING;
                Serial.flush();
            }
        }
    }
    if (state == PULSING)
    {
        display.setPartialWindow(0, 0, display.width(), display.height());

        display.firstPage();
        do
        {
            display.fillScreen(pulseColor ? GxEPD_BLACK : GxEPD_WHITE);
        } while (display.nextPage());

        pulseColor = !pulseColor;

        unsigned long startWait = millis();
        while (millis() - startWait < 1000)
        {
            yield();
            // If Python sends the length (and it's not a 'P' for PULSE)
            if (Serial.available() >= 4 && Serial.peek() != 'P')
            {
                state = WAIT_HEADER;
                break;
            }
        }
    }
    // 3. WAIT FOR HEADER (Length bytes)
    if (state == WAIT_HEADER)
    {
        if (Serial.available() >= 4)
        {
            // Peek to make sure we aren't accidentally reading 'PULSE' as a length
            if (Serial.peek() == 'P')
            {
                // This is a pulse command, not a length. Let the loop restart.
                return;
            }

            expectedLength = 0;
            for (int i = 0; i < 4; i++)
            {
                expectedLength = (expectedLength << 8) | Serial.read();
            }

            if (expectedLength > 0 && expectedLength <= MAX_DISPLAY_BUFFER_SIZE)
            {
                receivedBytes = 0;
                state = WAIT_IMAGE;
            }
        }
    }

    // 4. WAIT FOR IMAGE DATA (Your existing logic is mostly fine here)
    if (state == WAIT_IMAGE)
    {
        while (Serial.available() && receivedBytes < expectedLength)
        {
            imageBuffer[receivedBytes++] = Serial.read();
            yield();
        }

        Serial.flush();

        if (receivedBytes == expectedLength)
        {

            display.setFullWindow();
            display.firstPage();
            do
            {
                yield(); // CRITICAL on ESP8266
                display.fillScreen(GxEPD_WHITE);
                display.drawInvertedBitmap(
                    0, 0,
                    imageBuffer,
                    display.width(),
                    display.height(),
                    GxEPD_BLACK);
            } while (display.nextPage());

            display.powerOff();

            state = WAIT_HEADER; // Go back to waiting for the next cycle
            expectedLength = 0;
            receivedBytes = 0;
        }
    }
}
