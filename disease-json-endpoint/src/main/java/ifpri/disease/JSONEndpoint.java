package ifpri.disease;

import java.io.File;
import java.text.SimpleDateFormat;
import java.time.Month;
import java.time.format.TextStyle;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.fasterxml.jackson.databind.ObjectMapper;

@RestController
public class JSONEndpoint {

   private ObjectMapper objectMapper = new ObjectMapper();

   @Value("${conflicts.data}")
   private File dataDir;

   @GetMapping("/world")
   public Map world(@RequestParam(value = "lang", defaultValue = "en") String language,
         @RequestParam(value = "from", defaultValue = "") String from,
         @RequestParam(value = "to", defaultValue = "") String to) throws Exception {

      Calendar[] dates = validateDates(from, to);

      Map<String, Object> result = new LinkedHashMap<String, Object>();

      result.put("from", calToLong(dates[0]));
      result.put("to", calToLong(dates[1]));

      // this just means we can use before and include the date specified rather than
      // exclude the date. This is annoying but means we are consistent with the
      // previous endpoint behaviour
      dates[1].add(Calendar.DATE, 1);

      long docs = 0;
      Map<String, Long> documentsByDate = new LinkedHashMap<String, Long>();

      Map<String, Map> summary = new HashMap<String, Map>();

      while (dates[0].before(dates[1])) {

         String currentDate = calToLong(dates[0]).toString();

         documentsByDate.put(currentDate, 0L);

         File f = new File(dataDir, buildFilename(dates[0], language) + "/summary.json");

         dates[0].add(Calendar.DATE, 1);

         if (!f.exists())
            continue;

         List<Map<String, Object>> data = objectMapper.readValue(f, List.class);

         for (Map<String, Object> article : data) {

            // should this be the number of docs, or the number that contribute to the other
            // bits of data? Need to check with Soonho
            ++docs;
            documentsByDate.put(currentDate, documentsByDate.get(currentDate) + 1);

            List<Map<String, Object>> items = (List<Map<String, Object>>) article.get("items");

            for (Map<String, Object> item : items) {

               String disease = (String) ((Map) item.get("Disease")).get("Common_name");

               if (disease == null || "".equals(disease))
                  continue;

               disease = disease.toLowerCase();

               String country = (String) ((Map) item.get("Impacted_area")).get("Country");

               if (country == null || "".equals(country))
                  country = "OTHER";

               Map countryData = summary.getOrDefault(country, new LinkedHashMap());

               if (countryData.size() == 0) {
                  countryData.put("name", country);
                  countryData.put("articles", 0L);
                  countryData.put("diseases", new HashMap<String, Object>());
                  summary.put(country, countryData);
               }

               Map diseaseData = (Map) ((Map) countryData.get("diseases")).getOrDefault(disease,
                     new HashMap<String, Object>());

               if (diseaseData.size() == 0) {
                  diseaseData.put("name", disease);
                  diseaseData.put("occurances", 0L);
                  ((Map) countryData.get("diseases")).put(disease, diseaseData);
               }

               diseaseData.put("occurances", ((Long) diseaseData.get("occurances")) + 1);

            }
         }
      }

      result.put("countries", summary.values());

      return result;
   }

   @GetMapping("/country")
   public Map country(@RequestParam(value = "name", required = true) String country,
         @RequestParam(value = "lang", defaultValue = "en") String language,
         @RequestParam(value = "from", defaultValue = "") String from,
         @RequestParam(value = "to", defaultValue = "") String to) throws Exception {

      Calendar[] dates = validateDates(from, to);

      Map<String, Object> result = new LinkedHashMap<String, Object>();
      result.put("from", calToLong(dates[0]));
      result.put("to", calToLong(dates[1]));

      // this just means we can use before and include the date specified rather than
      // exclude the date. This is annoying but means we are consistent with the
      // previous endpoint behaviour
      dates[1].add(Calendar.DATE, 1);

      long docs = 0;
      Map<String, Long> documentsByDate = new LinkedHashMap<String, Long>();

      // this is where we will store info about the pages processed
      List<Map<String, Object>> pages = new ArrayList<Map<String, Object>>();

      while (dates[0].before(dates[1])) {

         String currentDate = calToLong(dates[0]).toString();

         documentsByDate.put(currentDate, 0L);

         File f = new File(dataDir, buildFilename(dates[0], language) + "/summary.json");

         dates[0].add(Calendar.DATE, 1);

         if (!f.exists())
            continue;

         List<Map<String, Object>> data = objectMapper.readValue(f, List.class);

         for (Map<String, Object> article : data) {

            // should this be the number of docs, or the number that contribute to the other
            // bits of data? Need to check with Soonho
            ++docs;
            documentsByDate.put(currentDate, documentsByDate.get(currentDate) + 1);

            // This will find and open the relevant plain text file to extract the matching
            // sentence, however, it seems we don't need to do this and can use
            // Orign_Sentence instead which is much quicker
            // String txtFileName = article.get("txt_file").toString();
            // txtFileName = txtFileName.substring(txtFileName.lastIndexOf('/') + 1);
            // File txtFile = new File(f.getParentFile(), txtFileName);
            // String txtFileContents = FileUtils.readFileToString(txtFile,"UTF-8");
            // String sentence = txtFileContents.substring(startOffset,endOffset);

            List<Map<String, Object>> items = (List<Map<String, Object>>) article.get("items");

            for (Map<String, Object> item : items) {

               String articleCountry = (String) ((Map) item.get("Impacted_area")).get("Country");

               if (articleCountry == null || "".equals(articleCountry))
                  articleCountry = "OTHER";

               // if the country of the article isn't the one we want then move on to the next
               // one in the summary file
               if (!country.equals(articleCountry))
                  continue;

               String disease = (String) ((Map) item.get("Disease")).get("Common_name");

               if (disease == null || "".equals(disease))
                  continue;

               disease = disease.toLowerCase();

               Map<String, Object> articleData = new LinkedHashMap<String, Object>();

               // fill in the basic article data
               articleData.put("url", article.get("url"));
               articleData.put("title", article.get("title").toString().replaceAll("</?title>", ""));
               articleData.put("date", Long.valueOf(currentDate));
               articleData.put("disease", disease);

               // int startOffset = (int) item.get("start_offset");
               // int endOffset = (int) item.get("end_offset");

               // NOTE: the spelling mistake here is intentional as it's wrong in the generated
               // JSON and so we need to match that to get any data out
               articleData.put("sentence", item.get("Orign_Sentence"));

               pages.add(articleData);
            }

         }

      }

      // finish putting the rest of the summary data into the map ready to be returned
      result.put("language", language);
      result.put("documents", docs);
      result.put("documentsByDate", documentsByDate);
      result.put("pages", pages);

      return result;
   }

   private Calendar[] validateDates(String from, String to) throws Exception {
      SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd");

      Calendar cTo = Calendar.getInstance();

      if (to == null || to.equals("")) {
         // no op as we are moving to inclusive of from but exclusive of to
      } else {
         cTo.setTime(dateFormat.parse(to));
      }

      Calendar cFrom = Calendar.getInstance();

      if (from == null || from.equals("")) {
         cFrom.setTimeInMillis(cTo.getTimeInMillis());
         cFrom.add(Calendar.DATE, -6);
      } else {
         cFrom.setTime(dateFormat.parse(from));
      }

      return new Calendar[] { cFrom, cTo };
   }

   private String buildFilename(Calendar c, String lang) {
      // TODO are the days of the month zero padded?

      return String.format("%02d", c.get(Calendar.DAY_OF_MONTH)) + "_"
            + Month.of(c.get(Calendar.MONTH) + 1).getDisplayName(TextStyle.SHORT, new Locale("en", "US")) + "_"
            + c.get(Calendar.YEAR);
   }

   /**
    * Convert a calendar instance to a long of the form yyyyMMdd
    */
   private Long calToLong(Calendar c) {
      return Long.valueOf(c.get(Calendar.YEAR) + String.format("%02d", (c.get(Calendar.MONTH) + 1))
            + String.format("%02d", c.get(Calendar.DAY_OF_MONTH)));
   }
}
