TARGET = main
LATEX  = pdflatex
SOURCES = $(TARGET).tex $(wildcard chapters/*.tex)

ifeq ($(OS),Windows_NT)
CLEAN_CMD = if exist "$(TARGET).aux" del /q "$(TARGET).aux" & if exist "$(TARGET).log" del /q "$(TARGET).log" & if exist "$(TARGET).out" del /q "$(TARGET).out" & if exist "$(TARGET).toc" del /q "$(TARGET).toc" & if exist "$(TARGET).pdf" del /q "$(TARGET).pdf"
else
CLEAN_CMD = rm -f $(TARGET).aux $(TARGET).log $(TARGET).out $(TARGET).toc $(TARGET).pdf
endif

.PHONY: all pdf clean

all: pdf

pdf: $(SOURCES)
	$(LATEX) -interaction=nonstopmode $(TARGET).tex
	$(LATEX) -interaction=nonstopmode $(TARGET).tex

clean:
	$(CLEAN_CMD)
