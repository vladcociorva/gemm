CC        ?= clang
CFLAGS    ?= -fPIC -std=c11
LIB_PATH  ?=
LDFLAGS   ?= -lopenblas

WARNFLAGS ?= -Wall -Wextra
CFLAGS    += $(WARNFLAGS)

SRC_DIR    = $(CURDIR)/src
BUILD_DIR  = $(CURDIR)/build
OBJ_DIR    = $(BUILD_DIR)/obj
BINARY     = runner 

# $(info )
# $(info Build info: )
# $(info CC: $(shell $(CC) --version))
# $(info CFLAGS: $(CFLAGS))
# $(info LDFLAGS: $(LDFLAGS))
# $(info )

MKDIR ?= @mkdir -p

ifdef OPENBLAS_PATH
	LIB_PATH += -L$(OBLAS_PATH)
endif

ifdef FAST
	CFLAGS += -O3
endif

default: build

runner.o: $(SRC_DIR)/runner.c
	$(MKDIR) $(OBJ_DIR)
	@$(CC) $(CFLAGS) -c $< -o $(OBJ_DIR)/$@

build: runner.o
	$(MKDIR) $(BUILD_DIR)
	@$(CC) $(CFLAGS) $(OBJ_DIR)/$< $(LIB_PATH) -o $(BUILD_DIR)/$(BINARY) $(LDFLAGS)

run: build
	@$(BUILD_DIR)/$(BINARY)

clean: 
	rm -rf $(BUILD_DIR)
