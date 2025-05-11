# flake8: noqa
import sys
import codecs


def create_tmx(en_file, pl_file, tmx_output):
    with codecs.open(en_file, "r", "utf-8") as en, codecs.open(pl_file, "r", "utf-8") as pl, codecs.open(
        tmx_output, "w", "utf-8"
    ) as tmx:
        # Write TMX header
        tmx.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        tmx.write('<!DOCTYPE tmx SYSTEM "tmx14.dtd">\n')
        tmx.write('<tmx version="1.4">\n')
        tmx.write('  <header creationtool="SimpleTMXCreator" creationtoolversion="1.0" ')
        tmx.write('segtype="sentence" o-tmf="PlainText" adminlang="en" srclang="en" datatype="plaintext">\n')
        tmx.write("  </header>\n")
        tmx.write("  <body>\n")

        # Read both files line by line simultaneously
        for en_line, pl_line in zip(en.readlines(), pl.readlines()):
            en_line = en_line.strip()
            pl_line = pl_line.strip()

            # Skip empty lines
            if not en_line or not pl_line:
                continue

            # Escape XML special characters
            en_line = en_line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            pl_line = pl_line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            # Write translation unit
            tmx.write("    <tu>\n")
            tmx.write(f'      <tuv xml:lang="en"><seg>{en_line}</seg></tuv>\n')
            tmx.write(f'      <tuv xml:lang="pl"><seg>{pl_line}</seg></tuv>\n')
            tmx.write("    </tu>\n")

        # Close TMX file
        tmx.write("  </body>\n")
        tmx.write("</tmx>\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python create_tmx.py english_file polish_file output_tmx")
        sys.exit(1)

    create_tmx(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"TMX file created: {sys.argv[3]}")
