#!/usr/bin/env bash
# T8 p4a: extrai de um zip da CTC só o PLY do frame inicial, o sparse do COLMAP desse
# frame e os .md5/licença da sequência para ~/mpeg/datasets/<seq>/, confere os md5
# internos e deixa a pasta somente leitura. O zip precisa já ter sido conferido contra a N420.
# Uso: prepare_sequence.sh <zip> <frame>   ex.: prepare_sequence.sh ~/mpeg/downloads/flowerdance.zip frame_0000
set -euo pipefail
ZIP=$(realpath "$1"); FRAME=$2
DS=${DATASETS:-$HOME/mpeg/datasets}
SEQ=$(basename "$ZIP" .zip)
DEST="$DS/$SEQ"

[[ -f "$ZIP" ]] || { echo "zip não encontrado: $ZIP"; exit 1; }
[[ "$FRAME" =~ ^frame_[0-9]{4}$ ]] || { echo "frame deve ser frame_NNNN"; exit 1; }
[[ -e "$DEST" ]] && { echo "já existe: $DEST (não sobrescrevo)"; exit 1; }

PLY="$SEQ/plys/$FRAME.ply"
SPARSE="$SEQ/colmap_data/$FRAME/sparse"
LIST=$(unzip -Z1 "$ZIP")
grep -qx "$PLY" <<<"$LIST" || { echo "PLY ausente no zip: $PLY"; exit 1; }
grep -q "^$SPARSE/" <<<"$LIST" || { echo "sparse ausente no zip: $SPARSE"; exit 1; }

mapfile -t MEMBROS < <(grep -E "^($PLY|$SPARSE/[^/]+|$SEQ/[^/]+\.(md5|md|pdf|txt))$" <<<"$LIST")
echo "== $SEQ $FRAME: ${#MEMBROS[@]} arquivos"; printf '   %s\n' "${MEMBROS[@]}"

chmod u+w "$DS"
TMP=$(mktemp -d "$DS/.tmp_${SEQ}_XXXX")
trap 'chmod -R u+w "$TMP" 2>/dev/null; rm -rf "$TMP"; chmod a-w "$DS"' EXIT
unzip -q "$ZIP" "${MEMBROS[@]}" -d "$TMP"

cd "$TMP/$SEQ"
cat ./*.md5 | tr -d '\r' | grep -E "\*?($SEQ/)?(plys/$FRAME\.ply|colmap_data/$FRAME/sparse/)" > /tmp/.md5_$$ || true
NL=$(wc -l < /tmp/.md5_$$)
NP=$(grep -c "plys/$FRAME\.ply" /tmp/.md5_$$ || true)
[[ $NP -eq 1 ]] || { echo "esperava 1 linha de md5 do PLY, achei $NP"; cat ./*.md5 | head; exit 1; }
sed -E "s#\*?$SEQ/#*#" /tmp/.md5_$$ | md5sum -c - ; rm -f /tmp/.md5_$$
NS=$(find "colmap_data/$FRAME/sparse" -type f | wc -l)
echo "md5 conferido: $NL linhas (PLY + $((NL-1)) do sparse; $NS arquivos no sparse)"
[[ $((NL-1)) -eq $NS ]] || echo "AVISO: nem todo arquivo do sparse tem md5 interno"
cd - >/dev/null

mv "$TMP/$SEQ" "$DEST"
chmod -R a-w "$DEST"
echo "pronto: $DEST (somente leitura)"
du -sh "$DEST"
