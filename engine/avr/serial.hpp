#ifndef AVR_SERIAL_H_
#define AVR_SERIAL_H_

/* Note: the SoftwareSerial library is not actually included with this header.
 * The user must explicitly include it before including this file.
 * 
 * Example:
 * ...
 * #include <SoftwareSerial.h>
 * ...
 * #include <avr/serial.hpp>
 * ...
 */

// Zhetapi namespace
namespace zhetapi {

namespace avr {

/* TODO: if these functions take up too much memory
 * then split the header to allow the user to select
 * only specific parts.
 */

// Type name
constexpr const char *type_name(int)
{
        return "int";
}

constexpr const char *type_name(long int)
{
        return "long int";
}

// End of line
struct endl_type {};

// Endl singleton
endl_type endl;

// TODO: add port here
class SerialInterface {
public:
        // Write functions
        virtual size_t write(uint8_t) = 0;
        virtual size_t write(char *) = 0;
        virtual size_t write(const char *) = 0;

        template <class T>
        friend SerialInterface &operator<<(SerialInterface &, const T &);
};

// Writing to serial
template <class T>
SerialInterface &operator<<(SerialInterface &serint, const T &value)
{
        serint.write("SerialInterface (#?): Unknown type...");

        return serint;
}

SerialInterface &operator<<(SerialInterface &serint, char *str)
{
        serint.write(str);

        return serint;
}

SerialInterface &operator<<(SerialInterface &serint, const char *str)
{
        serint.write(str);

        return serint;
}

template <>
SerialInterface &operator<<(SerialInterface &serint, const endl_type &el)
{
        serint.write(0x0D);
        serint.write(0x0A);

        return serint;
}

// General software serial
class UartSerial : public SerialInterface {
        SoftwareSerial  _ss;

        // For logging and information
        size_t          _rx     = 0;
        size_t          _tx     = 1;
        size_t          _port   = 9600;
public:
        UartSerial(size_t rx = 0, size_t tx = 1, size_t port = 9600)
                        :  _ss(rx, tx), _rx(rx),
                        _tx(tx), _port(port) {
                _ss.begin(_port);
        }

        virtual size_t write(uint8_t bv) override {
                return _ss.write(bv);
        }

        virtual size_t write(char *str) override {
                return _ss.write(str);
        }

        virtual size_t write(const char *str) override {
                return _ss.write(str);
        }

        template <class T>
        friend UartSerial &operator<<(UartSerial &, const T &);
};

// Wrapper around builtin Serial
class DefaultSerial : public SerialInterface {
        size_t  _port = 0;

        // Check and set default port
        void dinit_port() {
                if (!_port) {
                        _port = 9600;

                        Serial.begin(_port);
                }
        }
public:
        DefaultSerial() {}

        DefaultSerial(size_t port)
                        : _port(port) {
                Serial.begin(port);
        }

        virtual size_t write(uint8_t bv) override {
                dinit_port();
                return Serial.write(bv);
        }

        virtual size_t write(char *str) override {
                dinit_port();
                return Serial.write(str);
        }

        virtual size_t write(const char *str) override {
                dinit_port();
                return Serial.write(str);
        }

        template <class T>
        friend DefaultSerial &operator<<(DefaultSerial &, const T &);
};

// Operator<< overloads
template <class T>
DefaultSerial &operator<<(DefaultSerial &ds, const T &value)
{
        Serial.print("Serial (");
        Serial.print(ds._port);
        Serial.println(" baud): Unknown type...");
        return ds;
}

DefaultSerial &operator<<(DefaultSerial &ds, char *str)
{
        ds.write(str);

        return ds;
}

DefaultSerial &operator<<(DefaultSerial &ds, const char *str)
{
        ds.write(str);

        return ds;
}

template <>
DefaultSerial &operator<<(DefaultSerial &ds, const String &str)
{
        Serial.print(str);

        return ds;
}

template <>
DefaultSerial &operator<<(DefaultSerial &ds, const int &value)
{
        ds.dinit_port();
        Serial.print(value);
        return ds;
}

template <>
DefaultSerial &operator<<(DefaultSerial &ds, const long int &value)
{
        ds.dinit_port();
        Serial.print(value);
        return ds;
}

template <>
DefaultSerial &operator<<(DefaultSerial &ds, const double &value)
{
        ds.dinit_port();
        Serial.print(value);
        return ds;
}

template <>
DefaultSerial &operator<<(DefaultSerial &ds, const endl_type &el)
{
        ds.write(0x0D);
        ds.write(0x0A);

        return ds;
}

// Serial IO singleton
DefaultSerial serio;

}

}

#endif