from .CLSReaderBase import CLSReaderBase
import re
import pandas as pd


class IfpriConflictTrainReader(CLSReaderBase):
    def __init__(self, indomain_cls_input, outdomain_txt_input, update_target_labels=True, noteField='Notes', eventTypeField='Event Type', countryField='Country', locationField='Location', **kwargs):
        super().__init__(**kwargs)
        self.update_target_labels = update_target_labels
        if self.update_target_labels:
            if not self.target_labels:
                self.target_labels = []
        self.noteField = noteField
        self.eventTypeField = eventTypeField
        self.countryField = countryField
        self.locationField = locationField
        self._init_Reader(indomain_cls_input, outdomain_txt_input)
        if self.update_target_labels:
            print(self.target_labels)
        self._reset_iter()


    def _init_Reader(self, indomain_cls_input, outdomain_txt_input):
        self.all_ids = []
        self.data_dict = {}
        self.global_data_id = 0
        self.read_in_domain(indomain_cls_input)
        self.read_out_domain(outdomain_txt_input)

    def read_in_domain(self, indomain_cls_input):
        in_domain_csv_df = pd.read_csv(indomain_cls_input, quotechar='"', error_bad_lines=False)
        for index, row in in_domain_csv_df.iterrows():
            #current_note_text = row.NOTES
            #current_event_type = row.EVENT_TYPE
            #current_country  = row.COUNTRY
            #current_location = row.LOCATION

            current_note_text = row[self.noteField]
            current_event_type = row[self.eventTypeField]
            current_country  = row[self.countryField]
            current_location = row[self.locationField]
            
            if type(current_note_text) is str:
                if len(current_note_text) > 4:
                    self.all_ids.append(self.global_data_id)
                    self.data_dict[self.global_data_id] = {}
                    self.data_dict[self.global_data_id]['text']  = current_note_text
                    self.data_dict[self.global_data_id]['event'] = current_event_type
                    self.data_dict[self.global_data_id]['country'] = current_country
                    self.data_dict[self.global_data_id]['location'] = current_location
                    self.global_data_id += 1
                    if self.update_target_labels:
                        if (len(current_event_type) > 0) and (current_event_type not in self.target_labels):
                            self.target_labels.append(current_event_type)


    def read_out_domain(self, outdomain_txt_input):
        if self.update_target_labels:
            if 'NotConflict' not in self.target_labels:
                self.target_labels.append('NotConflict')
        with open(outdomain_txt_input, 'r') as fin:
            for line in fin:
                current_text = line.strip()
                if len(current_text) > 4:
                    self.all_ids.append(self.global_data_id)
                    self.data_dict[self.global_data_id] = {}
                    self.data_dict[self.global_data_id]['text']  = current_text
                    self.data_dict[self.global_data_id]['event'] = 'NotConflict'
                    self.data_dict[self.global_data_id]['country'] = 'Unknown'
                    self.data_dict[self.global_data_id]['location'] = 'Unknown'
                    self.global_data_id += 1
            




