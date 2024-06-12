from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("gemma-1.1-2b-it")
model = AutoModelForCausalLM.from_pretrained("gemma-1.1-2b-it")

def model_output(chat):
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs, max_new_tokens=500)
    
    bot_answer = tokenizer.decode(outputs[0])
    # print("MODEL OUTPUT:", bot_answer)
    start = "<start_of_turn>model"
    end = "<eos>"

    # Tìm vị trí bắt đầu của chuỗi con
    start_index = bot_answer.find(start) + len(start)
    # Tìm vị trí kết thúc của chuỗi con
    end_index = bot_answer.find(end)

    # Tách văn bản giữa hai vị trí đó
    result = bot_answer[start_index:end_index].strip()
    return result

# 1. Công ty con trực tiếp của FPT
subsidiaries_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin sau \
"Các Công ty con trực tiếp của FPT bao gồm: 
- CÔNG TY TNHH FPT DIGITAL
- CÔNG TY TNHH FPT SMART CLOUD
- CÔNG TY TNHH PHẦN MỀM FPT
- CÔNG TY HỆ THỐNG THÔNG TIN FPT
- CÔNG TY CỔ PHẦN VIỄN THÔNG FPT
- CÔNG TY CỔ PHẦN DỊCH VỤ TRỰC TUYẾN FPT
- CÔNG TY TNHH GIÁO DỤC FPT
- CÔNG TY TNHH ĐẦU TƯ FPT"
"""

# 2. Công ty liên kết trực tiếp với FPT
direct_affiliates_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin \
"Các Công ty liên kết trực tiếp với FPT bao gồm:
- CÔNG TY CỔ PHẦN BÁN LẺ KỸ THUẬT SỐ FPT
- CÔNG TY CỔ PHẦN SYNNEX FPT
Bên cạnh đó bạn có thể xem thêm chi tiết tại đây: https://fpt.com/vi/he-sinh-thai-fpt/cong-ty-thanh-vien"
"""

# 3. Các chương trình xã hội tại FPT
social_programs_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin \
"Các chương trình xã hội tại FPT
1. Kết nối cộng đồng hạnh phúc
- Chương trình Nâng bước em đến trường
- Hiến máu nhân đạo
2. Đầu tư cho giáo dục và thế hệ trẻ
- Học bổng cho các tài năng trẻ
- Học bổng Nguyễn Văn Đạo
- Trường Hy Vọng (Hope School)
- Chương trình Ánh sáng học đường và Chắp cánh ước mơ
- VioEdu - giải pháp Edtech 15 triệu tài khoản người dùng
- Violympic lan tỏa tình yêu toán học và đam mê công nghệ
3. Ươm mầm nhân ái
    Đóng góp tổng cộng 171 tỷ VNĐ cho cộng đồng
4. Bảo vệ môi trường
- Triển khai các giải pháp tiết kiệm năng lượng
- Tham gia bảo vệ động vật hoang dã
Bên cạnh đó bạn có thể xem thêm chi tiết về các chương trình trọng yếu của FPT tại đây: https://fpt.com/vi/ve-fpt/trach-nhiem-xa-hoi"
"""

# 4. Địa chỉ công ty FPT tại các quốc gia
addresses_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin sau \
"Địa chỉ của FPT tại VIỆT NAM như sau:
1. HANOI
- FPT Tower, số 10 Phố Phạm Văn Bạch, Phường Dịch Vọng, Quận Cầu Giấy, Hà Nội
- Điện thoại: +84 24 7300 7300
2. HO CHI MINH CITY
- Tòa nhà FPT Tân Thuận, Lô L29B -31B - 33B, đường Tân Thuận, KCX Tân Thuận, phường Tân Thuận Đông, Quận 7, Tp. Hồ Chí Minh
- Điện thoại: +84 28 7300 7300
 
Địa chỉ của FPT tại NHẬT BẢN như sau:
1. TOKYO 1
- Tokyo Head Quarter KDX Hamamatsucho Place 6F, Shibakoen 1-7-6, Minato-ku, Tokyo-to, 105-0011 Japan
- Điện thoại: +81-(3)-6634-6868
- Email: fjp.contact@fsoft.com.vn
2. TOKYO 2
- FPT Japanese Language School 4-3-5 Higashinippori, Arakawa-ku, Tokyo, 116-0014 Japan FPT Nippori Building
- Điện thoại: +81 3-5615-1012
- Email: fjp.contact@fsoft.com.vn
3. YOKOHAMA
- Pacific Marks Yokohama East 6F, Sakaecho 3-12, Kanagawa-ku, Yokohama-shi, Kanagawa-ken, 221-0052
- Điện thoại: +81-(45)-440-3738
- Email: fjp.contact@fsoft.com.vn
4. OSAKA
- JRE Dojima Tower Bld 16F, Dojima 2-4-27, Kita-ku, Osaka-shi, Osaka, 530-0003 Japan
- Điện thoại: +81-(6)-6344-9010
- Email: fjp.contact@fsoft.com.vn
5. FUKUOKA
- Daihakata Bld 2F, Hakataekimae 2-20-1, Hakata-ku, Fukuoka-shi 812-0011, Japan
- Điện thoại: + 81-(92)-452-9911
- Email: fjp.contact@fsoft.com.vn
6. HIROSHIMA
- Sankyo Hiroshima Bld 7F, Naka-ku, Komachi 3-25, Hiroshima-shi, Hiroshima-ken, 730-0041
- Điện thoại: +81-(82)-545-9806
- Email: fjp.contact@fsoft.com.vn
7. SAPPORO
- 7th floor Sapporo Tokeidai Building, Jonoshi 2-1 Kita 1 Chuo-ku, Sapporo-chi, Hokkaido 060-0001
- Điện thoại: +81-(11)-233-1266
- Email: fjp.contact@fsoft.com.vn
8. NAGOYA
- Nagoya Lucent Tower 11F, Ushijimacho 6-1, Nishi-ku, Nagoya-shi, Aichi 451-0046
- Điện thoại: +81-(52)-756-3456
- Email: fjp.contact@fsoft.com.vn
9. OKINAWA 1
- Okinawa R&D Center: JEI naha Building 6F 2-8-1 Kumoji, Naha-shi, Okinawa-ken 900-0015
- Điện thoại: +81-(98)-861-7311
- Email: fjp.contact@fsoft.com.vn
10. OKINAWA 2
- Okinawa Second R&D Center: COI Naha Bld.2F, Kume 2-3-15, Naha-shi, Okinawa-ken, 900-0033 Japan
- Điện thoại: +81-(98)-861-7311
- Email: support.japan@fpt-software.com
 

Địa chỉ của FPT tại TRUNG QUỐC như sau:
TRUNG QUỐC chỉ có 1 địa chỉ tại SHANGHAI
- 16th Floor, Global Creative Center, Lane 166, Minhong Road, Shanghai, P.R, China
- Điện thoại: +86 (21) 5416 2767
- Email: fcn.contact@fsoft.com.vn
 
Địa chỉ của FPT tại HÀN QUỐC như sau:
HÀN QUỐC chỉ có 1 địa chỉ tạiSEOUL
- 7F, Sewoo Building, Yeuidodong 10, Yeongdeungpogu, Seoul, Korea
- Điện thoại: +82 2 567 6650
- Email: fkr.contact@fsoft.com.vn
 

Địa chỉ của FPT tại HONGKONG như sau:
HONGKONG chỉ có 1 địa chỉ tại WANCHAI
- 17/F, Winsan Tower, 98 Thomson Road, Wanchai, Hong Kong
- Email: LinhNP1@fsoft.com.vn
 
Địa chỉ của FPT tại ĐÀI LOAN như sau:
ĐÀI LOAN chỉ có 1 địa chỉ tại 
- 9F-4, No.149-49 Keelung Road, Sec.2, Xinyi District, Taipei, Taiwan 11054, R.O.C
- Điện thoại: +886 2 27320741
- Email: Ted.Chiang@fsoft.com.vn
 
Địa chỉ của FPT tại LÀO như sau:
LÀO chỉ có 1 địa chỉ tại VIENTIANE
- 4th floor, Hatady Nuea Dits, Sailom road, Vientiane Plaza Hotel, Hatsady Neua village, Chanthabouly district, Vientiane, Laos
- Điện thoại: +856 207 771 0008
- Email: namHB2@fpt.com.vn
 
Địa chỉ của FPT tại THÁI LAN như sau:
THÁI LAN chỉ có 1 địa chỉ tại BANGKOK
- 35th Floor, Tower A, The Ninth Tower, 33/4, Rama 9 Road, Huaykwang District, 10310, Bangkok
- Điện thoại: +66 80 830 8997
- Email: LinhNP1@fsoft.com.vn
 
Địa chỉ của FPT tại PHILIPPINES như sau:
1. CEBU
- Ground Floor eBloc Tower 3, Geonzon Street, Cebu IT Park, Lahug Cebu City, Philippines 6000
- Điện thoại: +63 32 410 6857 (58)
- Email: support.philippines@fsoft.com.vn
2. CEBU 2
- 9th Floor HM Tower, West Geonzon Street, Cebu IT Park, Lahug Cebu City, Philippines 6000
- Điện thoại: +63 32 410 6857 (58)
- Email: support.philippines@fsoft.com.vn
3. MANILA
- Suite A, Level 14, Robinsons Summit Center, 6783 Ayala Avenue, Makati City, Philippines 1226
- Điện thoại: +63 32 410 6857 (58)
- Email: support.philippines@fsoft.com.vn
 
 - CAMPUCHIA
 PHNOM PHENH
 Building No.315, Vimean Canadia, 18th Floor, Corner Angduong street and Preah Munyvong street, Sangkat Vat Phnom, Khan Don Penh, Phnom Penh.
 
Địa chỉ của FPT tại MALAYSIA như sau:
MALAYSIA chỉ có 1 địa chỉ tại KUALAR LUMPUR
- Lot 19 – 01, Level 19 Menara Hapseng 2, 02, Jalan P. Ramlee, 50250 Kuala Lumpur, Malaysia
- Điện thoại: +603 2022 0333
- Email: support.malaysia@fpt-software.com
 
Địa chỉ của FPT tại SINGAPORE như sau:
SINGAPORE chỉ có 1 địa chỉ tại 
- 8 Kallang Avenue, 12-09 Aperia Tower 1, Singapore 339509
- Điện thoại: +65 6338 4353
- Email: support.singapore@fpt-software.com
 
Địa chỉ của FPT tại INDONESIA như sau:
INDONESIA chỉ có 1 địa chỉ tại JAKARTA
- Sovereign Plaza 6th Floor, JL TB Simatupang Kav.36, Jakarta 12430
- Điện thoại: (021) 2940-0239
- Email: support@fpt-software.com

Địa chỉ của FPT tại ẤN ĐỘ như sau:
ẤN ĐỘ chỉ có 1 địa chỉ tại HYDERABAD
- 1st floor ASR CREST, Plot No.42&43, Image Garden Road, Madhapur, Hyderabad – 500081, India
- Điện thoại: (091) 916 047 7345
- Email: findia.contact@fpt-software.com
 
Địa chỉ của FPT tại UAE như sau:
UAE chỉ có 1 địa chỉ tại DUBAI
- 1405, Fortune Tower, Jumeirah Lakes Towers, Dubai, UAE
- Điện thoại: +971 (04) 5776725
- Email: Support.Me@fpt-software.com
 
Địa chỉ của FPT tại ÚC như sau:
1. SYDNEY
- Level 45, 680 George Street, Sydney, NSW 2000, Australia
- Điện thoại: +61 2 9044 1350
2. MELBOURNE
- Level 3, 162 Collins St, Melbourne, VIC 3000
3. PERTH
- Level 11, Brookfield Place, 152 St. Georges Terrace

Địa chỉ của FPT tại ĐAN MẠCH như sau:
ĐAN MẠCH chỉ có 1 địa chỉ tại COPENHAGEN
- Diplomvej 381, DTU Sciencepark, DK-2800 Kgs. Lyngby
- Điện thoại: +45 316 753 48
- Email: pitt.sebens@fsoft.com.vn
 
Địa chỉ của FPT tại ANH như sau:
ANH chỉ có 1 địa chỉ tại LONDON
- FPT Software United Kingdom Ltd. 60 Cannon Street, London, EC4N 6NP
- Điện thoại: +020 45098064
- Email: support.uk@fpt-software.com
 
Địa chỉ của FPT tại ĐỨC như sau:
ĐỨC chỉ có 1 địa chỉ tại ESSEN
- FPT Deutschland GmbH: Am Thyssenhaus 1-3 (Haus 3), 45128 Essen, Germany
- Điện thoại: +49 201 4903 9350
- Email: support.germany@fpt-software.com
 
Địa chỉ của FPT tại PHÁP như sau:
1. PARIS
- 8 Terrasse Bellini, 92800 Puteaux, France
- Điện thoại: +33 018 0874 812
- Email: jerome.modolo@fsoft.com.vn
2. TOULOUSE
- Aeropole, Bat 1, 5 Avenue Albert Durand, 31700 Blagnac, France
- Điện thoại: +33 056 150 0437
- Email: jerome.modolo@fsoft.com.vn
 
Địa chỉ của FPT tại SLOVAKIA như sau:
SLOVAKIA chỉ có 1 địa chỉ tại KOŠICE
- FPT Slovakia S.R.O: Južná trieda 6, Košice 040 01, Slovakia
- Điện thoại: +421 55 610 16 20
- Email: fsvk.contact@fpt.sk
 
Địa chỉ của FPT tại BỈ như sau:
BỈ chỉ có 1 địa chỉ tại MECHELEN
- FPT Belgium: Schaliënhoevedreef 20h 2800 Mechelen, Belgium
- Email: contactus@fsoft.de
 
Địa chỉ của FPT tại CỘNG HÒA SÉC như sau:
CỘNG HÒA SÉC chỉ có 1 địa chỉ tại PRAHA-STRAŠNICE
- Fpt Czech S.R.O: Limuzská 3135/12, 108 00 Strašnice, Czech Republic
- Email: fsvk.contact@fpt.sk

Địa chỉ của FPT tại MỸ như sau:
1. TEXAS
- FPT USA Headquarters: 801 East Campbell Rd., Suite 525, Richardson, Texas 75081, USA
- Điện thoại: +1 214 253 2662
- Email: support.usa@fpt-software.com
2. LOS ANGELES
- 801 Parkview Drive North, Suite 100, El Segundo, CA 90245, USA
- Điện thoại: +1 424 336 9888
3. BOSTON
- 197 First Ave, Suite 200, Needham, MA 02494
- Điện thoại: +1 860 677 4427
4. HARTFORD
- 10 Stanford Drive, Farmington, CT 06032
- Điện thoại: +1 860 677 4427
5. DETROIT
- 17197 N Laurel Park Dr, Suite 273, Livonia, MI 48152
6. RENTON
- Renton Office: 901 Powell Avenue SW, Suite 111, Renton, WA 98057
- Điện thoại: + 1 650 931 7246
7. ATLANTA
- 2 Concourse Parkway, Suite 100, Atlanta, GA 30328
- Điện thoại: +1 404 442 8000
 
Địa chỉ của FPT tại COLOMBIA như sau:
COLOMBIA chỉ có 1 địa chỉ tại MEDELLIN
- Medellin, Colombia
- Email: support.usa@fpt-software.com
 
Địa chỉ của FPT tại CANADA như sau:
CANADA chỉ có 1 địa chỉ tại BOUCHERVILLE
- 242 Rue de Bayeux, Boucherville, QC J4B 7T9, Canada
- Điện thoại: +1 514 566 5658
- Email: support.canada@fpt-software.com
 
Địa chỉ của FPT tại COSTA RICA như sau:
COSTA RICA chỉ có 1 địa chỉ tại SAN JOSÉ
- San José, Costa Rica
- Email: support.usa@fpt-software.com"
"""

# 5. Đối tác 
partner_companies_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin sau \
"FPT hân hạnh trở thành đối tác với rất nhiều những công ty hàng đầu trên toàn thế giới, có thể kể đến như:
- Rheem
- Hitachi Solutions
- Tập đoàn Toshiba
- Denso Manufacturing
- Toppan Forms Operation
- A-Too Co., Ltd
- Tập đoàn Microsoft
- Amazon
- SAP
- Apple Authorized Reseller
Bạn có thể xem thêm chi tiết về các Đối tác và Khách hàng của FPT tài đây: https://fpt.com/vi/ve-fpt/doi-tac-va-khach-hang 
"
"""

# 6. Đồng phục
company_uniform_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin sau \
"Đồng phục của người nhà FPT tự hào có màu áo cam đặc trưng trong tất cả những sự kiện từ nhỏ đến lớp của Tập đoàn và các Công ty thành viên. Đồng phục của FPT được thiết kế trang trọng với áo polo cùng với logo FPT trước ngực, với những sự kiện lớp như: Kỷ niệm 25 năm thành lập, Kỷ niệm 35 năm thành lập... sẽ có thêm thiết kế chủ đề ở đằng sau lưng nhưng vẫn mang đậm chất FPT trong đó.
Ngoài áo đồng phục của Tập đoàn FPT, các Công ty thành viên cũng có thiết kế thêm nhiều mẫu đồng phục riêng rất sáng tạo, bắt mắt nhưng luôn toát ra tinh thần của nhà F.
Mời bạn xem chi tiết về lịch sử và văn hóa của sắc áo cam nhà F tại đây nha: https://chungta.vn/search.html?q=%C4%91%E1%BB%93ng+ph%E1%BB%A5c+fpt"
"""

# 7. Chiến lược phát triển
company_uniform_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin sau \
"Chiến lược phát triển bền vững của FPT được xây dựng dựa trên sự đảm bảo hài hòa của ba yếu tố:
- Phát triển kinh tế
- Hỗ trợ cộng đồng
- Bảo vệ môi trường
Do đó, cùng với việc đảm bảo sự tăng trưởng vững chắc về kinh tế, FPT cũng luôn chú trọng đến các hoạt động hỗ trợ cộng đồng dựa trên chính thế mạnh công nghệ của mình và đảm bảo mang đến những lợi ích tốt nhất cho các bên liên quan.
1. Cam kết với khách hàng:
- Cung cấp dịch vụ, giải pháp công nghệ tổng thể và toàn diện từ khâu tư vấn đến triển khai, vận hành, bảo trì.
- Mang lại những giá trị cao nhất cho khách hàng thông qua các sản phẩm, dịch vụ, giải pháp được phát triển dựa trên các xu hướng công nghệ mới.
- Không ngừng nâng cao uy tín thương hiệu của FPT.
- Xây dựng đội ngũ nhân sự chuyên nghiệp, có kinh nghiệm và năng lực chuyên môn cao.
2. Cam kết với cổ đông, nhà đầu tư:
- Đảm bảo lợi ích an toàn và bền vững cho cổ đông, nhà đầu tư.
- Cung cấp, cập nhật thông tin hai chiều kịp thời, đầy đủ và minh bạch với cổ đông
3. Cam kết với cán bộ nhân viên:
- Được tạo điều kiện và cơ hội phát huy cao nhất năng lực, nuôi dưỡng đam mê để thành công cùng Tập đoàn.
- Được đảm bảo các quyền lợi chính đáng cả về vật chất và tinh thần.
- Môi trường làm việc thân thiện, sáng tạo.
4. Cam kết cộng đồng:
- Điểm tựa tin cậy cho sự phát triển của cộng đồng.
- Mang lại những giá trị tốt đẹp hơn cho cuộc sống, tạo nên những giá trị bền vững thúc đẩy sự phát triển của xã hội, quốc gia.
5. Cam kết Chính phủ, Bộ, Ban, Ngành:
- Cam kết đồng hành với các chương trình, đề án lớn của Chương trình chuyển đổi số Quốc gia góp phần thúc đẩy phát triển kinh tế số, xã hội số, quốc gia số.
- Tuân thủ đầy đủ các quy định của ngành nói riêng và luật pháp nói chung.
- Hoàn thành tốt nhất nghĩa vụ đối với Nhà nước.
6. Cam kết với Đối tác, nhà cung cấp:
- Xây dựng quan hệ liên minh, cùng có lợi, đem lại thành công cho cả hai bên.
- Cùng mở rộng lĩnh vực kinh doanh, phát triển sản phẩm, dịch vụ mới.

Tầm nhìn và chiến lược FPT
1. Về kinh doanh:
- Với khách hàng là các doanh nghiệp lớn, Tập đoàn tập trung mở rộng/thúc đẩy cung cấp dịch vụ, giải pháp chuyển đổi số toàn diện từ khâu tư vấn đến triển khai. Trong đó, tập trung vào cung cấp các nền tảng, giải pháp công nghệ mới như RPA, Lowcode, AI, Blockchain… và các dịch vụ chuyển đổi, quản trị vận hành hạ tầng CNTT điện toán đám mây.
- Với khách hàng là các doanh nghiệp vừa và nhỏ, FPT tiếp tục phát triển mở rộng nhóm các giải pháp Made by FPT hướng tới một nền tảng quản trị duy nhất tất cả trong một và có khả năng kết nối mở rộng với các giải pháp, dịch vụ của bên thứ 3 nhằm tối ưu vận hành.
- Với khách hàng cá nhân, FPT mong muốn đem đến những trải nghiệm dịch vụ tốt nhất dựa trên các giải pháp và nền tảng quản trị mới.
2. Về công nghệ:
- FPT sẽ tập trung phát triển công nghệ theo hai hướng là phát triển các nền tảng, công nghệ lõi và gia tăng trải nghiệm khách hàng, hiệu quả vận hành dựa trên công nghệ.
- Trong đó, Tập đoàn sẽ tiếp tục đẩy mạnh nghiên cứu, phát triển chuyên sâu các giải pháp dựa trên công nghệ Blockchain, Lowcode, AI, Cloud cùng với các Nền tảng dữ liệu (Người dùng/Khách hàng/Dữ liệu nội bộ) đem lại các giải pháp kinh doanh hiệu quả, đáng tin cậy cho các tổ chức/tập đoàn lớn, doanh nghiệp vừa và nhỏ và những trải nghiệm đột phá cho khách hàng cá nhân.
3. Về con người:
- Tài sản lớn nhất của FPT là con người. Do đó, Tập đoàn luôn chú trọng xây dựng chính sách đãi ngộ theo hướng cạnh tranh, khuyến khích đổi mới, sáng tạo, đồng thời triển khai các chương trình đào tạo để xây đắp nên các thế hệ nhân viên không ngừng học hỏi và phấn đấu.
- Với triết lý đem lại cho mỗi thành viên điều kiện phát triển tài năng tốt nhất, FPT cam kết xây dựng một môi trường làm việc công bằng, minh bạch, không phân biệt đối xử. 
Mời bạn xem chi tiết tại đây nha: https://fpt.com/vi/ve-fpt/tam-nhin-chien-luoc"
"""

# 8. Giải thưởng đạt được
awards_achieved_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin sau \
"Trong hành trình xây dựng và phát triển, Tập đoàn FPT đã vinh dự đạt được rất nhiều giải thưởng trong nước và quốc tế, tiêu biểu có thể kể đến như:
- TOP 50 công ty niêm yết tốt nhất trong hơn 1 thập kỷ
- Doanh nghiệp CSR tiêu biểu của Việt Nam
- TOP các công ty nổi bật Châu Á
- Top “Strong Performer” về công nghệ RPA
- Giải Vàng và Top 10 sản phẩm số xuất sắc cho kinh tế số
Mời bạn xem thêm chi tiết tất cả giải thưởng của FPT tại đây: https://fpt.com/vi/ve-fpt/giai-thuong"
"""

# 9. Giá trị cốt lõi
core_values_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin sau \
"Giá trị cốt lõi của FPT là "Tôn, Đổi, Đồng - Chí, Gương, Sáng" được xem là một phần không thể thiếu kiến tạo nên bộ GEN của FPT, là tinh thần FPT, là sức mạnh thúc đẩy lãnh đạo, CBNV của Tập đoàn không ngừng nỗ lực, sáng tạo vì lợi ích chung của cộng đồng, khách hàng, cổ đông và các bên liên quan khác.
Tôn trọng - Đổi mới - Đồng đội - Chí công - Gương mẫu - Sáng suốt
Mời bạn xem thêm chi tiết tại đây: https://fpt.com/vi/ve-fpt/gia-tri-cot-loi"
"""

# 10. Sự kiện thường niên
annual_events_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin sau \
"Hội diễn Sao Chổi là sự kiện âm nhạc thường niên của người FPT. Hội diễn là điểm hẹn âm nhạc, sân chơi nghệ thuật lớn nhất của FPT, quy tụ những tiết mục độc đáo, tập luyện nghiêm túc và có sự đầu tư bài bản. Mỗi nhóm dự thi phải có tối thiểu 30 thành viên, thi theo hình thức live band, không sử dụng beat có sẵn. Thành viên ban nhạc có thể là người FPT hoặc nguồn lực từ bên ngoài. Để tăng tính độc đáo, các đội có thể sử dụng kết hợp nhiều loại nhạc cụ hoặc các hoạt động tạo ra âm thanh như gõ dép, chai, lọ... hay vỗ miệng.
Nếu cảm thấy “tò mò" về Hội diễn Sao Chổi của FPT, Chang mời bạn xem thêm chi tiết tại đây nhé:  https://chungta.vn/search.html?q=sao+ch%E1%BB%95i 

Hội làng là sự kiện truyền thống mang nét rất riêng của Tập đoàn mỗi dịp giáp Tết. Hội làng được tổ chức lần đầu tiên vào năm 1997 tại 89 Láng Hạ, Hà Nội. Thuở đó Hội làng được tổ chức 2 ngày 1 đêm. Ngày đầu tiên Lãnh đạo cùng CBNV nhà F thực hiện nghi thức cúng lễ, mổ lợn, gói bánh,... Sang ngày thứ 2 tất cả người nhà F sẽ cùng thực hiện dâng lễ cúng trời đất. Và cả làng FPT cùng ăn cỗ sau khi hết tuần hương. 
Sau 2 năm bị ảnh hưởng bởi đại dịch Covid-19, Hội làng FPT đã quay trở lại với người nhà F ngay tại sân chính của toà nhà FPT Tower. Mỗi con phố sẽ mang lại những trải nghiệm khác nhau từ trang phục, ẩm thực, trò chơi dân gian, câu chuyện lịch sử, văn hóa, con người... Mỗi năm, lễ hội sẽ có một chủ đề khác nhau và chính các Công ty thành viên sẽ là nơi sáng tạo từ chủ đề ấy. Những năm trở lại đây, Hội làng FPT được tổ chức 1 ngày duy nhất và kết hợp với lễ Rước trạng cực kì độc đáo.
Nếu thấy hứng thú với Hội làng của nhà F thì hãy tìm hiểu thêm ở link dưới đây nha:  https://chungta.vn/tag-31538.html

Rước trạng hay “Rước Trạng về làng” được Ban Văn hóa Đoàn thể FPT tổ chức, với sự tham gia của “Trưởng làng” Trương Gia Bình và Chủ lễ Nguyễn Văn Khoa. Thời gian tổ chức chương trình gần với Tết Nguyên Tiêu (Rằm tháng Giêng), hay còn gọi là Tết Thượng Nguyên hoặc Tết Trạng nguyên, ngay trong Hội làng. Xưa kia, đây là dịp các Trạng được hội họp để được đãi tiệc và mời vào vườn thượng uyển thăm hoa, ngắm cảnh, làm thơ. Với ý nghĩa này, chương trình sẽ tái hiện hoạt động rước trạng, trao sắc phong truyền thống của FPT. 
Để tìm hiểu thêm về nét văn hóa độc đáo này của nhà F, Chang mời bạn đọc các tin tức tại đây nha: https://chungta.vn/tag-37178.html 

Tập đoàn FPT tự hào là một trong những đơn vị dẫn đầu về Phong trào Văn hóa đoàn thể và các sự kiện nội bộ dành cho CBNV. Hằng năm, không chỉ những ngày lễ lớn như: Sinh nhật tập đoàn, Ngày vì cộng đồng... được đầu tư tổ chức, mà FPT còn lồng ghép những sự kiện nội bộ vào cả những ngày lễ nhỏ để người nhà F cảm nhận được rõ nhất văn hóa của công ty cũng như sự quan tâm, đầu tư dành cho môi trường và con người. 
Chang xin được giới thiệu tới bạn kênh thông tin lớn nhất và cập nhật nhanh nhất các sự kiện của Tập đoàn FPT tại đây nha: https://chungta.vn/
"
"""

# 11. Hội đồng quản trị
Board_of_Directors_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin sau \
"Hội đồng quản trị Tập đoàn FPT bao gồm:
- Ông TRƯƠNG GIA BÌNH - Chủ tịch Hội đồng quản trị
- Ông BÙI QUANG NGỌC - Phó Chủ tịch Hội đồng quản trị
- Ông ĐỖ CAO BẢO - Ủy viên Hội đồng Quản trị
- Ông JEAN-CHARLES BELLIOL - Ủy viên Hội đồng Quản trị Độc lập
- Ông HIROSHI YOKOTSUKA - Ủy viên Hội đồng Quản trị Độc lập
- Bà TRẦN THỊ HỒNG LĨNH - Ủy viên Hội đồng Quản trị 
Mời bạn xem thêm chi tiết về Hội đồng quản trị, Ban Kiểm soát, Ban Điều hành và đội ngũ giám đốc nghiệp vụ, Ban Điều hành công ty thành viên FPT tại đây: https://fpt.com/vi/ve-fpt/doi-ngu-lanh-dao"
"""

# 12. Các mốc thời gian đặc biệt
Special_Timeline_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin sau \
"Ngày 13/09 hằng năm là ngày Kỷ niệm thành lập công ty. Mỗi năm, FPT sẽ đều tổ chức Hội thao và Hội diễn tại các điểm cầu Hà Nội, TP Hồ Chí Minh, Cần Thơ, Đà Nẵng, Quy Nhơn. Những nét chấm phá đặc sắc, trải dài khắp Bắc - Trung - Nam luôn hứa hẹn đưa người F đi từ bất ngờ này đến bất ngờ khác. 

Ngày 13/9/1988, Viện trưởng Viên Nghiên cứu Công nghệ Quốc gia đã ký quyết định thành lập Công ty Công nghệ Thực phẩm (tên gọi đầu tiên của FPT) và giao cho ông Trương Gia Bình làm Giám đốc. Một công ty mới ra đời, không vốn liếng, không tài sản, không tiền mặt…, chỉ có 13 nhà khoa học trẻ tuổi, đầy hoài bão, tin tưởng vào bàn tay và trí óc của mình, dám đương đầu với mọi thách thức, quyết làm nên nghiệp lớn.
Năm 1990, FPT lựa chọn hướng kinh doanh tin học và có những hợp đồng phần mềm thương mại đầu tiên.
Nếu muốn tìm hiểu thêm về lịch sử phát triển của FPT, Chang mời bạn truy cập vào link dưới đây nhé: https://fpt.com/vi/ve-fpt/lich-su 

Sự kiện thành lập FPT
Ngày 13/9/1988, Viện trưởng Viên Nghiên cứu Công nghệ Quốc gia đã ký quyết định thành lập Công ty Công nghệ Thực phẩm (tên gọi đầu tiên của FPT) và giao cho ông Trương Gia Bình làm Giám đốc. Trụ sở đầu tiên của FPT được đặt tại 30 Hoàng Diệu, Hà Nội.
Ngày 13/4/2012, Chủ tịch HĐQT FPT Trương Gia Bình đã ký quyết định thành lập Hội đồng sáng lập FPT, gồm 13 thành viên:
- Trương Gia Bình
- Lê Quang Tiến
- Bùi Quang Ngọc
- Đỗ Cao Bảo
- Hoàng Minh Châu
- Trương Thị Thanh Thanh
- Nguyễn Thành Nam
- Hoàng Nam Tiến
- Trương Đình Anh
- Nguyễn Điệp Tùng
- Trần Quốc Hoài
- Lê Trường Tùng
- Phan Ngô Tống Hưng 
"
"""

# 13. Lĩnh vực kinh doanh
Business_Areas_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin sau \
"Hiện nay, FPT đang kinh doanh các lĩnh vực như:
1. Công nghệ:
FPT là công ty tiên phong trong các xu hướng công nghệ mới, đặc biệt là các công nghệ lõi của cuộc cách mạng 4.0, FPT đã nhanh chóng nắm bắt cơ hội từ thị trường toàn cầu để phát triển hệ sinh thái nền tảng, giải pháp, sản phẩm, dịch vụ Made by FPT giúp thúc đẩy nhanh quá trình chuyển đổi số của các doanh nghiệp, tổ chức trên thế giới.
Các giải pháp công nghệ của FPT bao gồm:
- Dịch vụ công nghệ thông tin
- Giải pháp theo ngành
- Tư vấn chuyển đổi số
- AI & điện toán đám mây
- Nền tảng quản trị doanh nghiệp
- Hệ sinh thái công nghệ Made-by-FPT
Với trên 35 năm kinh nghiệm chinh chiến tại 29 quốc gia và vùng lãnh thổ; hơn 28.000 kỹ sư, chuyên gia công nghệ của FPT sẽ mang tới cho khách hàng hơn 200 giải pháp công nghệ Made-by-FPT.
Mời bạn xem thêm chỉ tiết tại: https://fpt.com/vi/he-sinh-thai-fpt/cong-nghe
2. Viễn thông:
FPT là 1 trong 3 nhà cung cấp dịch vụ Internet hàng đầu Việt Nam, luôn theo sát các xu hướng thị trường và không ngừng nỗ lực đầu tư hạ tầng, nâng cấp chất lượng sản phẩm, dịch vụ, tăng cường ứng dụng công nghệ mới để mang đến cho khách hàng những trải nghiệm vượt trội. 
FPT không ngừng đầu tư, triển khai và tích hợp ngày càng nhiều các dịch vụ giá trị gia tăng trên cùng một đường truyền internet và kiến tạo hệ sinh thái truyền thông số:
- Internet FPT
- Truyền hình FPT
- Nhà thông minh
- Kênh thuê riêng
- Trung tâm dữ liệu
- Hệ sinh thái truyền thông số
Mời bạn xem thêm chỉ tiết tại: https://fpt.com/vi/he-sinh-thai-fpt/vien-thong
3. Giáo dục:
FPT là thương hiệu giáo dục có tầm ảnh hưởng quốc tế, Tổ chức Giáo dục FPT đã mở rộng đầy đủ các cấp học góp phần cung cấp nguồn nhân lực chất lượng cao cho thị trường lao động.
Hệ sinh thái của Tổ chức giáo dục FPT bao gồm:
- Đào tạo sau đại học và Đào tạo cho doanh nghiệp
- Đào tạo liên kết quốc tế
- Đào tạo đại học
- Đào tạo Cao đẳng
- Đào tạo CNTT trực tuyến FUNiX
- Hệ thống phổ thông FPT
Mời bạn xem thêm chi tiết tại: https://fpt.com/vi/he-sinh-thai-fpt/giao-duc"
"""

# 14. Logo
Logo_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin sau \
"Qua hơn 29 năm, FPT đã qua 2 lần thay đổi logo. Logo FPT đầu tiên - Tên thành viên sáng lập (năm 1988 - 1990) - logo FPT thứ hai - Nhiều màu sắc (năm 1991 - 13/9/2010) và logo FPT thứ ba - Hội tụ và kế thừa (13/9/2010 - Nay).
Logo FPT hiện tại có những nét cong dựa trên đường tròn hội tụ những tinh hoa FPT. Đường tròn thể hiện hình ảnh lan tỏa sức mạnh từ những ứng dụng mang đến cho cộng đồng. Những đường cong uyển chuyển liên tiếp, có xu hướng vươn lên, tựa như những ngọn lửa sinh khí mạnh mẽ luôn bừng lên đầy năng động.
Kiểu dáng 3 khối màu quen thuộc được tạo góc nghiêng 13 độ so với chiều thẳng đứng. Số 13 là con số linh thiêng luôn gắn bó với lịch sử thành lập và thành công của FPT - tạo cảm giác đi tới vững vàng.
Logo kế thừa và phát huy những giá trị cốt lõi của Thương hiệu FPT với 3 màu đặc trưng khá nổi bật. Màu cam được nhấn mạnh trong logo như sự ấm áp của mặt trời là màu tràn đầy sinh lực, năng động, trẻ trung và kích thích nhiệt huyết sáng tạo cho một thế giới tốt đẹp hơn. Màu cam cũng là màu thân thiện và cởi mở, thể hiện sự sẵn sàng chia sẻ, gắn kết trong cộng đồng.
Màu xanh lá cây trong logo bổ trợ cho ý nghĩa sức sống mạnh mẽ, hòa với tự nhiên. Đó là màu của sự thay đổi và phát triển. Màu xanh dương đậm đà là màu của năng lượng tự nhiên xuất phát từ vũ trụ. Màu cam tạo cảm giác mạnh mẽ và liên tưởng đến trí tuệ, tính bền vững và thống nhất.
Mời bạn đọc thêm về lịch sử, ý nghĩa và văn hóa sử dụng logo FPT tại đây nha: https://chungta.vn/search.html?q=logo+fpt "
"""

# 15. Xuất hiện tại các quốc gia
Appearance_in_countries_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin sau \
"FPT đã xuất hiện ở các nước châu Á, châu Âu và châu Mỹ:
1. Châu Á:
- VIỆT NAM
- NHẬT BẢN
- TRUNG QUỐC
- HÀN QUỐC
- LÀO
- THÁI LAN
- PHILIPPINES
- CAMPUCHIA
- MALAYSIA
- SINGAPORE
- INDONESIA
- ẤN ĐỘ
- UAE
- ÚC
2. Châu Âu:
- ĐAN MẠCH
- ANH
- ĐỨC
- PHÁP
- SLOVAKIA
- BỈ
- CỘNG HÒA SÉC
3. Châu Mỹ:
- MỸ
- COLOMBIA
- CANADA
- COSTA RICA"
"""

# 16. Thông tin tuyển dụng
Recruitment_Information_prompt = \
"""
Bạn là chatbot hỗ trợ hỏi đáp thông tin về công ty FPT. \
Hãy trả lời câu sau "{}" dựa trên thông tin sau \
"Trang web chính thức của Tập đoàn FPT là: https://fpt.com/vi. Đây là kênh thông tin chính thống giới thiệu về Tập đoàn FPT và là nơi cập nhật nhanh nhất những tin tức về Tập đoàn cũng như Các công ty thành viên. Mời bạn truy cập theo đường link phía trên để tìm hiểu ngay về FPt nhé!

Tập đoàn FPT là đơn vị tiên phong sự nghiệp đi đầu thị trường với các dự án giáo dục, công nghệ và viễn thông. FPT đã đạt danh hiệu “Top 1 Nơi làm việc tốt nhất ngành công nghệ” với văn hóa độc đáo, tinh thần “Leng Keng” đổi mới, không gian hiện đại, tôn trọng cá nhân và chăm sóc cho cán bộ nhân viên toàn diện. 
Mời bạn xem thông tin chi tiết tại đây nha: https://tuyendung.fpt.com/van-hoa"
"""

# INTENT 
intent_prompt = \
"""Phân loại câu sau "{}" thuộc lớp nào trong \
["CÔNG TY CON", "CÔNG TY LIÊN KẾT", "CÁC CHƯƠNG TRÌNH XÃ HỘI", "ĐỊA CHỈ CÔNG TY FPT TẠI CÁC QUỐC GIA",
"ĐỐI TÁC", "ĐỒNG PHỤC", "CHIẾN LƯỢC PHÁT TRIỂN", "GIẢI THƯỞNG ĐẠT ĐƯỢC",
"GIÁ TRỊ CỐT LÕI", "SỰ KIỆN THƯỜNG NIÊN", "HỘI ĐỒNG QUẢN TRỊ", "CÁC MỐC THỜI GIAN ĐẶC BIỆT",
"LĨNH VỰC KINH DOANH", "LOGO", "CÁC QUỐC GIA", "THÔNG TIN TUYỂN DỤNG"
].
Chỉ trả lời một tên lớp ở trên và không thêm gì khác cũng như " " vào câu trả lời
"""

while True:
    user_input = input("### User ###: ")
    if user_input == "/stop":
        break
    
    chat = [
        {
          "role": "user", 
          "content": intent_prompt.format(user_input)
        }
    ]
    
    intent = model_output(chat)
    print("### BOT ###:", intent)