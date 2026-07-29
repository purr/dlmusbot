"""Onboarding screen: /start, its language switcher, and the empty inline
help article.

This is the *only* translated surface in the bot — captions, status
buttons and error text stay English. Each language is one `Pack`; the
message is assembled from the pack plus whatever providers are actually
configured, so a Spotify-less deployment never claims Spotify support in
any language.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from providers.registry import Registry

from .status import example_inline_search_button
from .ui import esc

# Order for taglines and "X, Y, and Z" lists.
_ONBOARDING_ORDER: tuple[str, ...] = ("spotify", "soundcloud", "youtube_music")
_SERVICE_LABEL: dict[str, str] = {
    "spotify": "Spotify",
    "soundcloud": "SoundCloud",
    "youtube_music": "YouTube Music",
}

# URL kinds each service accepts. Only `track` is deliverable in DM; the
# rest are inline-only (see `_reject_collection` / `_reject_artist` in
# handlers/dm.py), which is what the container step explains.
_CONTAINER_KINDS: dict[str, tuple[str, ...]] = {
    "spotify": ("playlist", "album", "artist"),
    "soundcloud": ("playlist", "album", "artist"),
    "youtube_music": ("playlist",),
}
_CONTAINER_ORDER: tuple[str, ...] = ("playlist", "album", "artist")

# Real, long-lived links so the pasted example actually resolves. One per
# service; the message shows the one matching the configured providers.
# Spotify's is a spotify.link short link for a single track — verified
# live against core.shortlink.resolve.
_EXAMPLE_URLS: dict[str, str] = {
    "spotify": "https://spotify.link/73mfG0HF5Db",
    "soundcloud": "https://soundcloud.com/forss/flickermood",
    "youtube_music": "https://music.youtube.com/watch?v=kJQP7kiw5Fk",
}

# Seeds both the typed example and the switch-inline button below it.
EXAMPLE_QUERY = "drain gang"

LANG_CALLBACK_PREFIX = "start:lang:"


@dataclass(frozen=True)
class Pack:
    """One language of the onboarding screen.

    `sep` / `pair` / `last` are the list joiners: `pair` glues exactly two
    items, `last` glues the final item of three or more, `sep` the rest.
    `kinds` holds each container word already inflected for the slot it
    lands in inside `s_containers`.
    """

    label: str
    music: str
    perk_spotify: str
    perk_quality: str
    perk_tags: str
    s_type: str
    s_wait: str
    s_tap: str
    s_dm: str
    s_containers: str
    tip: str
    search: str
    kinds: dict[str, str]
    sep: str = ", "
    pair: str = " and "
    last: str = ", and "


LANGS: dict[str, Pack] = {
    "en": Pack(
        label="🇬🇧 English",
        music="Music",
        perk_spotify="🎵 Spotify audio comes straight from Spotify",
        perk_quality="✨ Best quality available, MP3 up to <b>320 kbps</b>",
        perk_tags="🆔 Proper ID3 tags and cover art on every file",
        s_type="Type the bot name, a space, then your search or link, "
        "for example:",
        s_wait="Wait a moment and a list opens above the textbox.",
        s_tap="Tap a result to download it. You get a copy here too.",
        s_dm="Or send a track link straight to this chat. Several links in one "
        "message are fine.",
        s_containers="{kinds} links are also supported.",
        tip="Tip: mute this bot and archive the chat to keep download "
        "notifications out of the way.",
        search="🔍 Click here to start searching",
        kinds={"playlist": "playlist", "album": "album", "artist": "artist"},
    ),
    "ru": Pack(
        label="🇷🇺 Русский",
        music="Музыка",
        perk_spotify="🎵 Аудио Spotify скачивается напрямую из Spotify",
        perk_quality="✨ Лучшее доступное качество, MP3 до <b>320 кбит/с</b>",
        perk_tags="🆔 Правильные ID3-теги и обложка в каждом файле",
        s_type="Напишите имя бота, пробел, потом запрос или ссылку, например:",
        s_wait="Подождите пару секунд, и над полем ввода откроется список.",
        s_tap="Нажмите на результат, чтобы скачать его. Копия придёт и сюда.",
        s_dm="Или пришлите ссылку на трек прямо в этот чат. Несколько ссылок в "
        "одном сообщении тоже работают.",
        s_containers="Ссылки на {kinds} тоже поддерживаются.",
        tip="Совет: отключите уведомления бота и заархивируйте чат, чтобы "
        "загрузки не мешали.",
        search="🔍 Нажмите, чтобы начать поиск",
        kinds={
            "playlist": "плейлисты",
            "album": "альбомы",
            "artist": "артистов",
        },
        pair=" и ",
        last=" и ",
    ),
    "fa": Pack(
        label="🇮🇷 فارسی",
        music="موسیقی",
        perk_spotify="🎵 صدای اسپاتیفای مستقیم از خود اسپاتیفای گرفته می‌شود",
        perk_quality="✨ بهترین کیفیت موجود، MP3 تا <b>320 کیلوبیت بر ثانیه</b>",
        perk_tags="🆔 تگ‌های ID3 درست و کاور روی هر فایل",
        s_type="نام ربات را بنویسید، یک فاصله، بعد عبارت جست‌وجو یا لینک، مثلاً:",
        s_wait="چند لحظه صبر کنید تا فهرست بالای کادر نوشتن باز شود.",
        s_tap="برای دانلود روی یک نتیجه بزنید. یک نسخه هم همین‌جا برایتان "
        "می‌آید.",
        s_dm="یا لینک آهنگ را مستقیم در همین چت بفرستید. چند لینک در یک پیام هم "
        "مشکلی ندارد.",
        s_containers="لینک‌های {kinds} هم پشتیبانی می‌شود.",
        tip="نکته: این ربات را بی‌صدا کنید و چت را آرشیو کنید تا اعلان‌های "
        "دانلود مزاحم نشوند.",
        search="🔍 برای شروع جست‌وجو اینجا بزنید",
        kinds={
            "playlist": "پلی‌لیست",
            "album": "آلبوم",
            "artist": "هنرمند",
        },
        sep="، ",
        pair=" و ",
        last=" و ",
    ),
    "zh": Pack(
        label="🇨🇳 中文",
        music="音乐",
        perk_spotify="🎵 Spotify 音频直接从 Spotify 下载",
        perk_quality="✨ 最佳音质，MP3 最高 <b>320 kbps</b>",
        perk_tags="🆔 每个文件都带完整 ID3 标签和封面",
        s_type="输入机器人名字，空一格，再打关键词或链接，例如：",
        s_wait="稍等片刻，结果列表会出现在输入框上方。",
        s_tap="点一个结果即可下载。这里也会收到一份。",
        s_dm="或者把单曲链接直接发到这个聊天里。一条消息里发多个链接也可以。",
        s_containers="{kinds}链接同样支持。",
        tip="提示：把机器人静音并归档聊天，下载通知就不会打扰你。",
        search="🔍 点这里开始搜索",
        kinds={"playlist": "歌单", "album": "专辑", "artist": "艺人"},
        sep="、",
        pair="和",
        last="和",
    ),
    "es": Pack(
        label="🇪🇸 Español",
        music="Música",
        perk_spotify="🎵 El audio de Spotify se descarga directamente de Spotify",
        perk_quality="✨ La mejor calidad disponible, MP3 hasta <b>320 kbps</b>",
        perk_tags="🆔 Etiquetas ID3 correctas y carátula en cada archivo",
        s_type="Escribe el nombre del bot, un espacio y luego tu búsqueda o "
        "enlace, por ejemplo:",
        s_wait="Espera un momento y se abre una lista encima del cuadro de "
        "texto.",
        s_tap="Toca un resultado para descargarlo. Aquí también te llega una "
        "copia.",
        s_dm="O envía el enlace de una canción directamente a este chat. Varios "
        "enlaces en un mismo mensaje también valen.",
        s_containers="Los enlaces de {kinds} también son compatibles.",
        tip="Consejo: silencia este bot y archiva el chat para que las "
        "notificaciones de descarga no molesten.",
        search="🔍 Toca aquí para buscar",
        kinds={"playlist": "playlists", "album": "álbumes", "artist": "artistas"},
        pair=" y ",
        last=" y ",
    ),
    "ar": Pack(
        label="🇸🇦 العربية",
        music="موسيقى",
        perk_spotify="🎵 صوت Spotify يُنزَّل مباشرة من Spotify",
        perk_quality="✨ أفضل جودة متاحة، MP3 حتى <b>320 كيلوبت/ث</b>",
        perk_tags="🆔 وسوم ID3 صحيحة وصورة غلاف لكل ملف",
        s_type="اكتب اسم البوت، ثم مسافة، ثم كلمة البحث أو الرابط، مثلاً:",
        s_wait="انتظر لحظة وستفتح قائمة فوق مربع الكتابة.",
        s_tap="اضغط على نتيجة لتنزيلها. تصلك نسخة هنا أيضًا.",
        s_dm="أو أرسل رابط أغنية مباشرة إلى هذه المحادثة. عدة روابط في رسالة "
        "واحدة مقبولة أيضًا.",
        s_containers="روابط {kinds} مدعومة أيضًا.",
        tip="نصيحة: اكتم هذا البوت وأرشِف المحادثة حتى لا تزعجك إشعارات "
        "التنزيل.",
        search="🔍 اضغط هنا لبدء البحث",
        kinds={
            "playlist": "قوائم التشغيل",
            "album": "الألبومات",
            "artist": "الفنانين",
        },
        sep=" و",
        pair=" و",
        last=" و",
    ),
    "hi": Pack(
        label="🇮🇳 हिन्दी",
        music="संगीत",
        perk_spotify="🎵 Spotify का ऑडियो सीधे Spotify से आता है",
        perk_quality="✨ सबसे अच्छी उपलब्ध क्वालिटी, MP3 <b>320 kbps</b> तक",
        perk_tags="🆔 हर फ़ाइल पर सही ID3 टैग और कवर आर्ट",
        s_type="बॉट का नाम लिखें, एक स्पेस दें, फिर अपना सर्च या लिंक लिखें, जैसे:",
        s_wait="थोड़ा रुकें, टेक्स्टबॉक्स के ऊपर लिस्ट खुल जाती है।",
        s_tap="डाउनलोड करने के लिए किसी नतीजे पर टैप करें। एक कॉपी यहाँ भी आ "
        "जाती है।",
        s_dm="या गाने का लिंक सीधे इसी चैट में भेजें। एक ही मैसेज में कई लिंक भी "
        "चलते हैं।",
        s_containers="{kinds} लिंक भी सपोर्ट किए जाते हैं।",
        tip="टिप: इस बॉट को म्यूट करें और चैट आर्काइव कर दें ताकि डाउनलोड "
        "नोटिफिकेशन परेशान न करें।",
        search="🔍 सर्च शुरू करने के लिए यहाँ टैप करें",
        kinds={
            "playlist": "प्लेलिस्ट",
            "album": "एल्बम",
            "artist": "आर्टिस्ट",
        },
        pair=" और ",
        last=" और ",
    ),
    "pt": Pack(
        label="🇧🇷 Português",
        music="Música",
        perk_spotify="🎵 O áudio do Spotify vem direto do Spotify",
        perk_quality="✨ Melhor qualidade disponível, MP3 até <b>320 kbps</b>",
        perk_tags="🆔 Tags ID3 corretas e capa em todos os arquivos",
        s_type="Digite o nome do bot, um espaço e depois sua busca ou link, "
        "por exemplo:",
        s_wait="Espere um instante e uma lista abre acima da caixa de texto.",
        s_tap="Toque num resultado para baixá-lo. Uma cópia chega aqui também.",
        s_dm="Ou mande o link de uma faixa direto neste chat. Vários links na "
        "mesma mensagem também funcionam.",
        s_containers="Links de {kinds} também são suportados.",
        tip="Dica: silencie este bot e arquive a conversa para as notificações "
        "de download não atrapalharem.",
        search="🔍 Toque aqui para buscar",
        kinds={
            "playlist": "playlists",
            "album": "álbuns",
            "artist": "artistas",
        },
        pair=" e ",
        last=" e ",
    ),
    "tr": Pack(
        label="🇹🇷 Türkçe",
        music="Müzik",
        perk_spotify="🎵 Spotify sesi doğrudan Spotify'dan indirilir",
        perk_quality="✨ Mevcut en iyi kalite, <b>320 kbps</b>'ye kadar MP3",
        perk_tags="🆔 Her dosyada doğru ID3 etiketleri ve kapak görseli",
        s_type="Bot adını yazın, boşluk bırakın, sonra aramanızı veya "
        "bağlantıyı ekleyin, örneğin:",
        s_wait="Biraz bekleyin, yazı kutusunun üstünde bir liste açılır.",
        s_tap="İndirmek için bir sonuca dokunun. Buraya da bir kopya gelir.",
        s_dm="Ya da parça bağlantısını doğrudan bu sohbete gönderin. Tek "
        "mesajda birkaç bağlantı da olur.",
        s_containers="{kinds} bağlantıları da destekleniyor.",
        tip="İpucu: Bu botu sessize alın ve sohbeti arşivleyin, indirme "
        "bildirimleri rahatsız etmesin.",
        search="🔍 Aramaya başlamak için dokunun",
        kinds={
            "playlist": "çalma listesi",
            "album": "albüm",
            "artist": "sanatçı",
        },
        pair=" ve ",
        last=" ve ",
    ),
    "id": Pack(
        label="🇮🇩 Indonesia",
        music="Musik",
        perk_spotify="🎵 Audio Spotify diunduh langsung dari Spotify",
        perk_quality="✨ Kualitas terbaik yang ada, MP3 hingga <b>320 kbps</b>",
        perk_tags="🆔 Tag ID3 lengkap dan sampul di setiap file",
        s_type="Ketik nama bot, beri spasi, lalu kata kunci atau tautanmu, "
        "misalnya:",
        s_wait="Tunggu sebentar, daftar hasil muncul di atas kotak teks.",
        s_tap="Ketuk salah satu hasil untuk mengunduhnya. Salinannya juga masuk "
        "ke sini.",
        s_dm="Atau kirim tautan lagu langsung ke chat ini. Beberapa tautan "
        "dalam satu pesan juga boleh.",
        s_containers="Tautan {kinds} juga didukung.",
        tip="Tips: bisukan bot ini dan arsipkan chat supaya notifikasi unduhan "
        "tidak mengganggu.",
        search="🔍 Ketuk di sini untuk mulai mencari",
        kinds={"playlist": "playlist", "album": "album", "artist": "artis"},
        pair=" dan ",
        last=", dan ",
    ),
}

# Button grid, 3 per row. The active language's slot shows English instead,
# so switching back is always one tap and the grid keeps its shape.
_LANG_GRID: tuple[str, ...] = (
    "ru", "fa", "zh",
    "es", "ar", "hi",
    "pt", "tr", "id",
)
_ROW_WIDTH = 3


def pack_for(lang: str) -> Pack:
    return LANGS.get(lang) or LANGS["en"]


def visible_help_providers(registry: Registry, cfg: Any) -> frozenset[str]:
    """Which services to name in user-facing help.

    SoundCloud: no config, include if registered.
    Spotify: include if registered (requires ``SP_DC`` at build time).
    YouTube Music: include only if registered and ``YT_COOKIES_FILE`` is set
    (help text matches a "fully configured" YT setup; the provider may
    still work for public tracks without cookies).
    """
    names = set(registry.names())
    out: set[str] = set()
    if "soundcloud" in names:
        out.add("soundcloud")
    if "spotify" in names:
        out.add("spotify")
    cookies = (getattr(cfg, "YT_COOKIES_FILE", None) or "").strip()
    if "youtube_music" in names and cookies:
        out.add("youtube_music")
    return frozenset(out)


def _ordered_labels(visible: frozenset[str]) -> list[str]:
    return [_SERVICE_LABEL[k] for k in _ONBOARDING_ORDER if k in visible]


def _join(parts: list[str], p: Pack) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]}{p.pair}{parts[1]}"
    return p.sep.join(parts[:-1]) + p.last + parts[-1]


def _tagline(visible: frozenset[str], p: Pack) -> str:
    labels = _ordered_labels(visible)
    return "<b>" + (" · ".join(labels) if labels else p.music) + "</b>"


def _example_url(visible: frozenset[str]) -> str:
    for name in _ONBOARDING_ORDER:
        if name in visible:
            return _EXAMPLE_URLS[name]
    return _EXAMPLE_URLS["spotify"]


def _container_step(visible: frozenset[str], p: Pack) -> str:
    kinds = {k for name in visible for k in _CONTAINER_KINDS.get(name, ())}
    words = [p.kinds[k] for k in _CONTAINER_ORDER if k in kinds]
    if not words:
        return ""
    line = p.s_containers.format(kinds=_join(words, p))
    # Sentence-cases scripts that have case; a no-op for the rest.
    return line[0].upper() + line[1:]


def format_onboarding_card_description(visible: frozenset[str]) -> str:
    """One-line summary for the empty inline-query result card."""
    parts: list[str] = []
    if "spotify" in visible:
        parts.append("Spotify")
    if "soundcloud" in visible:
        parts.append("SC")
    if "youtube_music" in visible:
        parts.append("YT")
    head = " · ".join(parts) if parts else "Music"
    return f"{head} · inline or DM · up to 320 kbps"


def format_onboarding_message(
    username: str, visible_providers: frozenset[str], lang: str = "en"
) -> str:
    """Rich text for <code>/start</code> and the empty inline help article (HTML)."""
    p = pack_for(lang)
    perks: list[str] = []
    if "spotify" in visible_providers:
        perks.append(p.perk_spotify)
    perks.append(p.perk_quality)
    perks.append(p.perk_tags)
    mention = f"@{esc(username)}"
    examples = (
        f"  <code>{mention} {EXAMPLE_QUERY}</code>\n"
        f"  <code>{mention} {_example_url(visible_providers)}</code>"
    )
    rest = [p.s_wait, p.s_tap, p.s_dm, _container_step(visible_providers, p)]
    steps = f"• {p.s_type}\n{examples}\n" + "\n".join(f"• {s}" for s in rest if s)
    return "\n\n".join([_tagline(visible_providers, p), "\n".join(perks), steps])


def format_start_message(
    username: str, visible_providers: frozenset[str], lang: str = "en"
) -> str:
    p = pack_for(lang)
    return (
        format_onboarding_message(username, visible_providers, lang)
        + "\n\n"
        + f"<i>{p.tip}</i>"
    )


def format_inline_empty_message(username: str, visible_providers: frozenset[str]) -> str:
    return format_onboarding_message(username, visible_providers)


def start_kb(lang: str = "en") -> InlineKeyboardMarkup:
    """/start keyboard: the language grid, then the inline-search primer."""
    codes = ["en" if c == lang else c for c in _LANG_GRID]
    rows = [
        [
            InlineKeyboardButton(
                text=LANGS[c].label,
                callback_data=f"{LANG_CALLBACK_PREFIX}{c}",
            )
            for c in codes[i:i + _ROW_WIDTH]
        ]
        for i in range(0, len(codes), _ROW_WIDTH)
    ]
    rows.append(
        [example_inline_search_button(EXAMPLE_QUERY, text=pack_for(lang).search)]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)
