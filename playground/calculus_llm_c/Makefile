CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude
LDFLAGS = -lm
SRC = src/math_c.c src/tokenizer_c.c src/positronic_c.c main.c
OBJ = $(SRC:.c=.o)
TARGET = calculus_c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) $(OBJ)

.PHONY: all clean
